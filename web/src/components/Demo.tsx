import { useEffect, useMemo, useRef, useState } from "react";
import { copy, type Lang } from "../i18n";
import { emaBlend, fkcnToGrimace, fkcnWeights } from "../emotion/fkcn";
import { FaceAnimator } from "../emotion/interpolator";
import { apiBase, checkApi, streamChat, type ChatTurn } from "../engine/api";
import { GrimaceFace } from "./GrimaceFace";

type TokenSpan = {
  text: string;
  valence?: number;
  arousal?: number;
  dominance?: number;
};

type Round = {
  user: string;
  tokens: TokenSpan[];
};

type ConnStatus = "loading" | "ready" | "error";

const REST_FACE = {
  joy: 0,
  surprise: 0,
  fear: 0,
  sadness: 0,
  disgust: 0,
  anger: 0,
};

function formatTip(tok: TokenSpan): string | undefined {
  if (tok.valence === undefined || tok.arousal === undefined) return undefined;
  if (tok.dominance === undefined) {
    return `V:${tok.valence.toFixed(2)} A:${tok.arousal.toFixed(2)}`;
  }
  return `V:${tok.valence.toFixed(2)} A:${tok.arousal.toFixed(2)} D:${tok.dominance.toFixed(2)}`;
}

function systemPrompt(lang: Lang): string {
  return lang === "zh"
    ? "请简洁回复，一两句话即可。不要啰嗦，少用表情符号。"
    : "Reply in one or two short sentences. No preamble. Avoid emojis.";
}

export function Demo({ lang }: { lang: Lang }) {
  const t = copy[lang];
  const [conn, setConn] = useState<ConnStatus>("loading");
  const [connErr, setConnErr] = useState("");
  const [attempt, setAttempt] = useState(0);
  const [input, setInput] = useState("");
  const [rounds, setRounds] = useState<Round[]>([]);
  const [busy, setBusy] = useState(false);
  const [face, setFace] = useState(REST_FACE);
  const abortRef = useRef<AbortController | null>(null);
  const animator = useMemo(() => new FaceAnimator(0.42, 6), []);
  const emaRef = useRef<number[]>([0, 0, 0, 0, 0, 0, 0]);
  const historyRef = useRef<Round[]>([]);
  const tokenQueueRef = useRef<TokenSpan[]>([]);
  const drainTokensRef = useRef<Promise<void>>(Promise.resolve());

  useEffect(() => {
    historyRef.current = rounds;
  }, [rounds]);

  useEffect(() => {
    let cancelled = false;
    void checkApi().then((ok) => {
      if (cancelled) return;
      if (ok) {
        setConn("ready");
        setConnErr("");
      } else {
        setConn("error");
        setConnErr(`${apiBase()} unreachable`);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [attempt]);

  useEffect(() => {
    let raf = 0;
    const tick = () => {
      const stepped = animator.step();
      if (stepped) {
        setFace({
          joy: stepped[0],
          surprise: stepped[1],
          fear: stepped[2],
          sadness: stepped[3],
          disgust: stepped[4],
          anger: stepped[5],
        });
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [animator]);

  function retryConn() {
    setConn("loading");
    setConnErr("");
    setAttempt((n) => n + 1);
  }

  async function drainTokenQueue() {
    while (tokenQueueRef.current.length > 0) {
      const span = tokenQueueRef.current.shift();
      if (!span) break;

      if (span.valence != null && span.arousal != null) {
        const raw = fkcnWeights(span.valence, span.arousal);
        emaRef.current = emaBlend(emaRef.current, raw, 0.78);
        const g = fkcnToGrimace(emaRef.current);
        animator.setTarget([g.joy, g.surprise, g.fear, g.sadness, g.disgust, g.anger]);
        await animator.waitUntilSettled({ threshold: 0.055, timeoutMs: 120 });
      }

      setRounds((prev) => {
        const copyRounds = prev.map((round) => ({
          ...round,
          tokens: [...round.tokens],
        }));
        const last = copyRounds[copyRounds.length - 1];
        last.tokens.push(span);
        return copyRounds;
      });
    }
  }

  function enqueueToken(span: TokenSpan) {
    tokenQueueRef.current.push(span);
    drainTokensRef.current = drainTokensRef.current.then(() => drainTokenQueue());
  }

  async function send(text: string) {
    const msg = text.trim();
    if (!msg || busy || conn !== "ready") return;
    setInput("");
    setBusy(true);
    tokenQueueRef.current = [];
    drainTokensRef.current = Promise.resolve();
    emaRef.current = [0, 0, 0, 0, 0, 0, 0];
    animator.snapToTarget();
    setRounds((prev) => [...prev, { user: msg, tokens: [] }]);

    const messages: ChatTurn[] = [{ role: "system", content: systemPrompt(lang) }];
    for (const round of historyRef.current) {
      messages.push({ role: "user", content: round.user });
      messages.push({
        role: "assistant",
        content: round.tokens.map((tok) => tok.text).join(""),
      });
    }
    messages.push({ role: "user", content: msg });

    const controller = new AbortController();
    abortRef.current = controller;
    try {
      await streamChat(
        messages,
        { maxNewTokens: 64, temperature: 0.4, signal: controller.signal },
        (piece) => {
          enqueueToken({
            text: piece.text,
            valence: piece.valence,
            arousal: piece.arousal,
            dominance: piece.dominance,
          });
        },
      );
      await drainTokensRef.current;
      animator.snapToTarget();
    } catch (err) {
      if (!controller.signal.aborted) {
        setConn("error");
        setConnErr(err instanceof Error ? err.message : String(err));
      }
    } finally {
      abortRef.current = null;
      setBusy(false);
    }
  }

  function stop() {
    abortRef.current?.abort();
  }

  const canSend = conn === "ready" && !busy;

  return (
    <section className="demo" id="demo">
      <div className="section-head">
        <h2>{t.demoTitle}</h2>
        <p className="muted">{t.demoHint}</p>
      </div>
      {conn === "loading" ? <p className="status muted">{t.loadConnecting}</p> : null}
      {conn === "error" ? (
        <p className="status status-error">
          {`${t.loadError} ${connErr}`.trim()}{" "}
          <button type="button" className="linkish" onClick={retryConn}>
            {t.loadRetry}
          </button>
        </p>
      ) : null}
      <div className="demo-grid">
        <div className="chat-card">
          <div className="transcript" aria-live="polite">
            {rounds.length === 0 ? <p className="empty">{t.emptyChat}</p> : null}
            {rounds.map((round, i) => (
              <div className="round" key={`${round.user}-${i}`}>
                <div className="bubble user">
                  <span className="who">{t.you}</span>
                  {round.user}
                </div>
                <div className="bubble bot">
                  <span className="who">{t.model}</span>
                  {round.tokens.map((tok, j) => {
                    const tip = formatTip(tok);
                    return (
                      <span
                        className={tip ? "tok" : undefined}
                        data-tip={tip}
                        key={`${i}-${j}-${tok.text}`}
                      >
                        {tok.text}
                      </span>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
          <form
            className="composer"
            onSubmit={(e) => {
              e.preventDefault();
              void send(input);
            }}
          >
            <textarea
              rows={2}
              value={input}
              placeholder={canSend ? t.placeholder : t.placeholderLoading}
              disabled={conn !== "ready" || busy}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  void send(input);
                }
              }}
            />
            {busy ? (
              <button type="button" onClick={stop}>
                {t.stop}
              </button>
            ) : (
              <button type="submit" disabled={!canSend}>
                {t.send}
              </button>
            )}
          </form>
          <div className="examples">
            <span>{t.examplesLabel}</span>
            {t.examples.map((ex) => (
              <button key={ex} type="button" disabled={!canSend} onClick={() => void send(ex)}>
                {ex}
              </button>
            ))}
          </div>
        </div>
        <aside className="face-card">
          <GrimaceFace weights={face} />
        </aside>
      </div>
    </section>
  );
}
