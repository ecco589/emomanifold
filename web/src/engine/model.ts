import {
  configureTransformersEnv,
  modelHostLabel,
  NETWORK_STALL_MS,
  pickModelHost,
  WASM_INIT_STALL_MS,
} from "./transformersEnv";

export type LoadProgress = {
  status: "checking" | "need-gpu" | "loading" | "ready" | "error";
  phase?: "tokenizer" | "onnx" | "init";
  file?: string;
  progress?: number;
  loadedBytes?: number;
  totalBytes?: number;
  message?: string;
  host?: string;
};

function asRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" ? (value as Record<string, unknown>) : null;
}

function tensorData(tensor: unknown): { data: ArrayLike<number>; dims: number[] } | null {
  const rec = asRecord(tensor);
  if (!rec) return null;
  const data = rec.data as ArrayLike<number> | undefined;
  const dims = rec.dims as number[] | undefined;
  if (!data || !dims) return null;
  return { data, dims };
}

function lastRow(tensor: unknown): Float32Array | null {
  const parsed = tensorData(tensor);
  if (!parsed) return null;
  const { data, dims } = parsed;
  const last = dims[dims.length - 1];
  if (!last) return null;
  const offset = data.length - last;
  const out = new Float32Array(last);
  for (let i = 0; i < last; i += 1) {
    out[i] = Number(data[offset + i]);
  }
  return out;
}

function argmax(values: ArrayLike<number>): number {
  let best = 0;
  let bestVal = -Infinity;
  for (let i = 0; i < values.length; i += 1) {
    const v = values[i];
    if (v > bestVal) {
      bestVal = v;
      best = i;
    }
  }
  return best;
}

function sample(values: ArrayLike<number>, temperature: number): number {
  if (temperature <= 1e-5) return argmax(values);
  let maxLog = -Infinity;
  for (let i = 0; i < values.length; i += 1) {
    const v = values[i] / temperature;
    if (v > maxLog) maxLog = v;
  }
  const exps = new Float64Array(values.length);
  let sum = 0;
  for (let i = 0; i < values.length; i += 1) {
    const e = Math.exp(values[i] / temperature - maxLog);
    exps[i] = e;
    sum += e;
  }
  let r = Math.random() * sum;
  for (let i = 0; i < exps.length; i += 1) {
    r -= exps[i];
    if (r <= 0) return i;
  }
  return exps.length - 1;
}

function pickHidden(output: Record<string, unknown>): Float32Array | null {
  const direct = output.last_hidden_state ?? output.hidden_states;
  if (direct) {
    if (Array.isArray(direct)) {
      return lastRow(direct[direct.length - 1]);
    }
    const row = lastRow(direct);
    if (row) return row;
  }
  for (const [key, value] of Object.entries(output)) {
    if (key === "logits" || key.includes("past") || key.includes("present")) continue;
    if (key.includes("hidden")) {
      const row = lastRow(value);
      if (row && row.length > 8 && row.length < 8192) return row;
    }
  }
  return null;
}

export type ChatTurn = { role: "system" | "user" | "assistant"; content: string };

export type GeneratedToken = {
  text: string;
  hidden: Float32Array | null;
};

type TransformersMod = typeof import("@huggingface/transformers");

let tokenizer: Awaited<ReturnType<TransformersMod["AutoTokenizer"]["from_pretrained"]>> | null = null;
let model: Awaited<ReturnType<TransformersMod["AutoModelForCausalLM"]["from_pretrained"]>> | null = null;
let hiddenAvailable: boolean | null = null;

export function didExposeHiddenStates(): boolean | null {
  return hiddenAvailable;
}

const MODEL_ID = "onnx-community/Qwen2.5-0.5B-Instruct";

let loadPromise: Promise<void> | null = null;
let loadCallbacks = new Set<(p: LoadProgress) => void>();
let lastProgressState: LoadProgress = { status: "loading", progress: 0 };

function emitProgress(p: LoadProgress) {
  lastProgressState = p;
  for (const cb of loadCallbacks) cb(p);
}

function withStallWatch<T>(
  work: (touch: () => void) => Promise<T>,
  stallMs: number,
  label: string,
): Promise<T> {
  let last = Date.now();
  const touch = () => {
    last = Date.now();
  };
  touch();

  const stallReject = new Promise<never>((_, reject) => {
    const id = window.setInterval(() => {
      if (Date.now() - last > stallMs) {
        window.clearInterval(id);
        reject(new Error(`${label}: no response for ${Math.round(stallMs / 1000)}s`));
      }
    }, 500);
  });

  return Promise.race([work(touch), stallReject]);
}

function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = window.setTimeout(() => {
      reject(new Error(`${label} timed out after ${Math.round(ms / 1000)}s`));
    }, ms);
    promise.then(
      (v) => {
        window.clearTimeout(timer);
        resolve(v);
      },
      (err) => {
        window.clearTimeout(timer);
        reject(err);
      },
    );
  });
}

export function isModelReady(): boolean {
  return model !== null && tokenizer !== null;
}

export function resetModelLoad(): void {
  loadPromise = null;
  tokenizer = null;
  model = null;
  hiddenAvailable = null;
  lastProgressState = { status: "loading", progress: 0 };
}

export async function loadOnDeviceModel(
  onProgress: (p: LoadProgress) => void,
): Promise<void> {
  loadCallbacks.add(onProgress);
  if (isModelReady()) {
    onProgress({ status: "ready", progress: 100 });
    loadCallbacks.delete(onProgress);
    return;
  }
  onProgress(lastProgressState);
  if (!loadPromise) {
    loadPromise = loadOnDeviceModelOnce().catch((err) => {
      loadPromise = null;
      emitProgress({
        status: "error",
        message: err instanceof Error ? err.message : String(err),
      });
      throw err;
    });
  }
  try {
    await loadPromise;
    if (isModelReady()) {
      onProgress({ status: "ready", progress: 100 });
    }
  } finally {
    loadCallbacks.delete(onProgress);
  }
}

async function loadTokenizer(
  AutoTokenizer: TransformersMod["AutoTokenizer"],
  host: string,
  touch: () => void,
) {
  let lastProgress = 0;
  return AutoTokenizer.from_pretrained(MODEL_ID, {
    progress_callback: (info: {
      status?: string;
      file?: string;
      progress?: number;
      loaded?: number;
      total?: number;
    }) => {
      touch();
      let progress = info.progress;
      if (progress == null && info.loaded != null && info.total) {
        progress = (100 * info.loaded) / info.total;
      }
      if (info.status === "download") progress = 0;
      if (progress == null) progress = lastProgress;
      lastProgress = progress;
      emitProgress({
        status: "loading",
        phase: "tokenizer",
        host,
        file: info.file,
        progress: Math.min(15, (progress / 100) * 15),
        loadedBytes: info.loaded,
        totalBytes: info.total,
      });
    },
  } as never);
}

async function loadOnnxModel(
  AutoModelForCausalLM: TransformersMod["AutoModelForCausalLM"],
  host: string,
  dtype: string,
  touch: () => void,
) {
  let lastProgress = 0;
  return AutoModelForCausalLM.from_pretrained(MODEL_ID, {
    device: "wasm",
    dtype,
    progress_callback: (info: {
      status?: string;
      file?: string;
      progress?: number;
      loaded?: number;
      total?: number;
    }) => {
      touch();
      let progress = info.progress;
      if (progress == null && info.loaded != null && info.total) {
        progress = (100 * info.loaded) / info.total;
      }
      if (info.status === "download") progress = 0;
      if (progress == null) progress = lastProgress;
      lastProgress = progress;
      const scaled = Math.min(88, 15 + Math.min(73, (progress / 100) * 73));
      emitProgress({
        status: "loading",
        phase: progress >= 99 && info.file?.includes(".onnx") ? "init" : "onnx",
        host,
        file: info.file,
        progress: scaled,
        loadedBytes: info.loaded,
        totalBytes: info.total,
      });
    },
  } as never);
}

async function loadOnDeviceModelOnce(): Promise<void> {
  emitProgress({ status: "loading", progress: 0, phase: "init" });

  const host = await pickModelHost((h) => {
    emitProgress({ status: "loading", progress: 0, phase: "init", host: h });
  });

  if (!host) {
    const message = "Cannot reach model CDN. Check network and retry.";
    emitProgress({ status: "error", message });
    throw new Error(message);
  }

  const { env, AutoTokenizer, AutoModelForCausalLM } = await import("@huggingface/transformers");
  if (env.backends?.onnx?.wasm) {
    env.backends.onnx.wasm.numThreads = 1;
  }
  configureTransformersEnv(env, host);

  let lastError: unknown;

  try {
    tokenizer = await withStallWatch(
      (touch) => loadTokenizer(AutoTokenizer, host, touch),
      NETWORK_STALL_MS,
      "Tokenizer",
    );
  } catch (err) {
    lastError = err;
    tokenizer = null;
  }

  if (!tokenizer) {
    const message = formatLoadError(lastError, host);
    emitProgress({ status: "error", host, message });
    throw new Error(message);
  }

  for (const dtype of ["q4"]) {
    try {
      emitProgress({ status: "loading", progress: 15, host, phase: "onnx" });
      model = await withStallWatch(async (touch) => {
        const started = Date.now();
        const heartbeat = window.setInterval(() => {
          touch();
          const elapsed = Date.now() - started;
          const est = Math.min(88, 15 + (elapsed / 240_000) * 73);
          emitProgress({
            status: "loading",
            phase: elapsed > 120_000 ? "init" : "onnx",
            progress: est,
            host,
          });
        }, 2000);
        try {
          return await loadOnnxModel(AutoModelForCausalLM, host, dtype, touch);
        } finally {
          window.clearInterval(heartbeat);
        }
      }, WASM_INIT_STALL_MS, `ONNX ${dtype}`);
      lastError = null;
      break;
    } catch (err) {
      lastError = err;
      model = null;
    }
  }

  if (!model) {
    const message = formatLoadError(lastError, host);
    emitProgress({ status: "error", host, message });
    throw new Error(message);
  }

  emitProgress({ status: "ready", host, progress: 100 });
}

function formatLoadError(err: unknown, host: string): string {
  const raw = err instanceof Error ? err.message : String(err);
  const source = modelHostLabel(host);
  if (/failed to fetch|networkerror|load failed|no response|timed out|abort/i.test(raw)) {
    return `${raw} — ${source} unreachable`;
  }
  return raw;
}

export async function generateStream(
  messages: ChatTurn[],
  opts: {
    maxNewTokens?: number;
    temperature?: number;
    signal?: AbortSignal;
  },
  onToken: (token: GeneratedToken) => void,
): Promise<void> {
  if (!model || !tokenizer) {
    throw new Error("Model is not loaded");
  }
  const maxNewTokens = opts.maxNewTokens ?? 64;
  const temperature = opts.temperature ?? 0.4;

  const encoded = tokenizer.apply_chat_template(messages, {
    add_generation_prompt: true,
    tokenize: true,
    return_dict: true,
  } as never) as Record<string, unknown>;

  const { Tensor } = await import("@huggingface/transformers");
  let inputIds = encoded.input_ids;
  const attentionMask = encoded.attention_mask;
  let pastKeyValues: unknown;
  const eos = Number(tokenizer.eos_token_id ?? 151645);

  for (let step = 0; step < maxNewTokens; step += 1) {
    if (opts.signal?.aborted) break;

    const forwardInput: Record<string, unknown> = pastKeyValues
      ? { input_ids: inputIds, past_key_values: pastKeyValues }
      : { input_ids: inputIds, attention_mask: attentionMask };

    const output = (await (model as unknown as (x: unknown) => Promise<unknown>)(forwardInput)) as Record<
      string,
      unknown
    >;

    const logits = lastRow(output.logits);
    if (!logits) break;
    const nextId = sample(logits, temperature);
    if (nextId === eos) break;

    const hidden = pickHidden(output);
    if (hiddenAvailable === null) hiddenAvailable = hidden !== null;
    else if (hidden) hiddenAvailable = true;

    const piece = tokenizer.decode([nextId], { skip_special_tokens: false }) as string;
    onToken({ text: piece, hidden });

    pastKeyValues = output.past_key_values ?? output.present;
    inputIds = new Tensor("int64", BigInt64Array.from([BigInt(nextId)]), [1, 1]);
  }
}
