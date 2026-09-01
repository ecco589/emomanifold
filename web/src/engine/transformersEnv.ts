type TransformersEnv = {
  remoteHost: string;
  allowRemoteModels: boolean;
  useBrowserCache: boolean;
};

export const MODEL_REPO = "onnx-community/Qwen2.5-0.5B-Instruct";
const PROBE_FILE = "tokenizer_config.json";

const LAB_HOST = "http://ubuntu.tail1701c2.ts.net:8787/";
const HF_HOST = "https://huggingface.co/";

export const WASM_INIT_STALL_MS = 45_000;
export const NETWORK_STALL_MS = 12_000;

function normalizeHost(raw: string): string {
  return raw.endsWith("/") ? raw : `${raw}/`;
}

/** Host candidates in priority order (comma-separated VITE_MODEL_HOST overrides). */
export function modelHosts(): string[] {
  const override = import.meta.env.VITE_MODEL_HOST as string | undefined;
  if (override?.trim()) {
    return [...new Set(override.split(",").map((h) => normalizeHost(h.trim())).filter(Boolean))];
  }
  if (import.meta.env.PROD) return [HF_HOST];
  return [LAB_HOST];
}

export function modelFileUrl(host: string, file: string): string {
  const base = host.endsWith("/") ? host : `${host}/`;
  return `${base}${MODEL_REPO}/resolve/main/${file}`;
}

export async function probeModelHost(host: string, timeoutMs = NETWORK_STALL_MS): Promise<boolean> {
  const url = modelFileUrl(host, PROBE_FILE);
  const ac = new AbortController();
  const timer = window.setTimeout(() => ac.abort(), timeoutMs);
  try {
    const res = await fetch(url, { signal: ac.signal, cache: "no-store" });
    return res.ok;
  } catch {
    return false;
  } finally {
    window.clearTimeout(timer);
  }
}

export async function pickModelHost(onTrying?: (host: string) => void): Promise<string | null> {
  for (const host of modelHosts()) {
    onTrying?.(host);
    if (await probeModelHost(host)) return host;
  }
  return null;
}

export function configureTransformersEnv(env: TransformersEnv, host: string): string {
  env.remoteHost = host.endsWith("/") ? host : `${host}/`;
  env.allowRemoteModels = true;
  env.allowLocalModels = false;
  env.useBrowserCache = true;
  return env.remoteHost;
}

export function modelHostLabel(host: string): string {
  if (host.includes("tail") || host.includes("100.70.217.19")) return "lab server";
  if (host.includes("huggingface")) return "Hugging Face";
  if (host.includes("hf-mirror")) return "HF Mirror";
  return "model CDN";
}
