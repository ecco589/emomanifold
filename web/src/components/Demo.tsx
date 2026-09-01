import { useEffect, useMemo, useRef, useState } from "react";
import { copy, type Lang } from "../i18n";
import { emaBlend, fkcnToGrimace, fkcnWeights } from "../emotion/fkcn";
import { FaceAnimator } from "../emotion/interpolator";
import { projectHidden, type PipelineConfig } from "../emotion/projector";
import {
  didExposeHiddenStates,
  generateStream,
  isModelReady,
  loadOnDeviceModel,
  resetModelLoad,
  type ChatTurn,
  type LoadProgress,
} from "../engine/model";
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
  const [load, setLoad] = useState<LoadProgress>({ status: "loading", progress: 0 });
  const [loadAttempt, setLoadAttempt] = useState(0);
  const [pipeline, setPipeline] = useState<PipelineConfig | null>(null);
  const [input, setInput] = useState("");
  const [rounds, setRounds] = useState<Round[]>([]);
  const [busy, setBusy] = useState(false);
  const [face, setFace] = useState(REST_FACE);
  const [hiddenWarning, setHiddenWarning] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const animator = useMemo(() => new FaceAnimator(10, 0.08, 6), []);
  const emaRef = useRef<number[]>([0, 0, 0, 0, 0, 0, 0]);
  const historyRef = useRef<Round[]>([]);

  useEffect(() => {
    historyRef.current = rounds;
  }, [rounds]);

  useEffect(() => {
    let raf = 0;
    const tick = (now: number) => {
      const stepped = animator.step(now);
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

  useEffect(() => {
    let cancelled = false;

    fetch("./pipeline/pipeline.json")
      .then((r) => r.json())
      .then((data: PipelineConfig) => {
        if (!cancelled) setPipeline(data);
      })
      .catch(() => {
        if (!cancelled) setLoad({ status: "error", message: "Missing pipeline.json" });
      });

    void loadOnDeviceModel((p) => {
      if (cancelled) return;
      setLoad(p);
    })
      .then(() => {
        if (!cancelled && isModelReady()) {
          setLoad({ status: "ready", progress: 100 });
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setLoad({
            status: "error",
            message: err instanceof Error ? err.message : String(err),
          });
        }
      });

    return () => {
      cancelled = true;
    };
  }, [loadAttempt]);

  function retryLoad() {
    resetModelLoad();
    setLoad({ status: "loading", progress: 0 });
    setLoadAttempt((n) => n + 1);
  }

  const loadPct =
    load.status === "loading"
      ? Math.min(90, Math.max(0, Math.round(load.progress ?? 0)))
      : null;
  const loadHint =
    load.status === "loading" && (load.progress ?? 0) >= 85
      ? t.loadInitializing
      : load.phase === "tokenizer"
        ? t.loadPhaseTokenizer
        : load.phase === "onnx"
          ? t.loadPhaseOnnx
          : null;

  async function send(text: string) {
    const msg = text.trim();
    if (!msg || busy || !(load.status === "ready" || isModelReady())) return;
    setInput("");
    setBusy(true);
    setHiddenWarning(false);
    emaRef.current = [0, 0, 0, 0, 0, 0, 0];
    const nextRounds: Round[] = [...historyRef.current, { user: msg, tokens: [] }];
    setRounds(nextRounds);

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
      await generateStream(
        messages,
        { maxNewTokens: 64, temperature: 0.4, signal: controller.signal },
        (piece) => {
          const span: TokenSpan = { text: piece.text };
          if (piece.hidden && pipeline) {
            const vad = projectHidden(piece.hidden, pipeline);
            span.valence = vad.valence;
            span.arousal = vad.arousal;
            span.dominance = vad.dominance;
            const raw = fkcnWeights(vad.valence, vad.arousal);
            emaRef.current = emaBlend(emaRef.current, raw, 0.6);
            const g = fkcnToGrimace(emaRef.current);
            animator.setTarget([g.joy, g.surprise, g.fear, g.sadness, g.disgust, g.anger]);
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
        },
      );
      if (didExposeHiddenStates() === false) setHiddenWarning(true);
    } catch (err) {
      setLoad({
        status: "error",
        message: err instanceof Error ? err.message : String(err),
      });
    } finally {
      abortRef.current = null;
      setBusy(false);
    }
  }

  function stop() {
    abortRef.current?.abort();
  }

  const canSend = (load.status === "ready" || isModelReady()) && !busy;
  const canType = load.status === "loading" || load.status === "ready";
  const inputPlaceholder = canSend ? t.placeholder : t.placeholderLoading;

  return (
    <section className="demo" id="demo">
      <div className="section-head">
        <h2>{t.demoTitle}</h2>
        <p className="muted">{t.demoHint}</p>
      </div>
      {loadPct !== null ? (
        <div className="load-bar" aria-label={`${loadPct}%`}>
          <div className="load-bar-track">
            <div className="load-bar-fill" style={{ width: `${loadPct}%` }} />
          </div>
          <span className="load-bar-pct">{loadPct}%</span>
          {loadHint ? <span className="load-bar-hint muted">{loadHint}</span> : null}
        </div>
      ) : null}
      {load.status === "need-gpu" ? <p className="status status-need-gpu">{t.loadNeedGpu}</p> : null}
      {load.status === "error" ? (
        <p className="status status-error">
          {`${t.loadError} ${load.message ?? ""}`.trim()}{" "}
          <button type="button" className="linkish" onClick={retryLoad}>
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
              placeholder={inputPlaceholder}
              disabled={!canType}
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
      {hiddenWarning ? <p className="status status-error">{t.hiddenMissing}</p> : null}
    </section>
  );
}
