export type ApiToken = {
  text: string;
  valence?: number;
  arousal?: number;
  dominance?: number;
};

export type ChatTurn = { role: "system" | "user" | "assistant"; content: string };

const DEFAULT_API = "http://ubuntu.tail1701c2.ts.net:8788";

export function apiBase(): string {
  const raw = (import.meta.env.VITE_API_URL as string | undefined) ?? DEFAULT_API;
  return raw.replace(/\/$/, "");
}

export async function checkApi(timeoutMs = 8000): Promise<boolean> {
  const ac = new AbortController();
  const timer = window.setTimeout(() => ac.abort(), timeoutMs);
  try {
    const res = await fetch(`${apiBase()}/health`, { signal: ac.signal });
    if (!res.ok) return false;
    const body = (await res.json()) as { ok?: boolean };
    return body.ok === true;
  } catch {
    return false;
  } finally {
    window.clearTimeout(timer);
  }
}

export async function streamChat(
  messages: ChatTurn[],
  opts: { maxNewTokens?: number; temperature?: number; signal?: AbortSignal },
  onToken: (t: ApiToken) => void,
): Promise<void> {
  const res = await fetch(`${apiBase()}/v1/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages,
      max_new_tokens: opts.maxNewTokens ?? 64,
      temperature: opts.temperature ?? 0.4,
    }),
    signal: opts.signal,
  });
  if (!res.ok || !res.body) {
    throw new Error(`API ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const parts = buf.split("\n\n");
    buf = parts.pop() ?? "";
    for (const block of parts) {
      const line = block.split("\n").find((l) => l.startsWith("data: "));
      if (!line) continue;
      const data = line.slice(6);
      if (data === "[DONE]") return;
      const payload = JSON.parse(data) as ApiToken;
      onToken(payload);
    }
  }
}
