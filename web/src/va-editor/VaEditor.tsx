import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { GrimaceFace } from "../components/GrimaceFace";
import { emaBlend, fkcnToGrimace, fkcnWeights } from "../emotion/fkcn";
import { parseVaCsv, toVaCsv, type VaRow } from "./csv";

type Series = "v" | "a";

const params = new URLSearchParams(window.location.search);
const COND = params.get("cond") ?? "emomanifold";
const TRIAL = params.get("trial") ?? "004";
/** Same alpha as RAL/experiment/C/C1/code/render_grimace_from_source_va.py. */
const EMA_ALPHA = 0.6;

const STEP_X = 22;
const PAD = { left: 44, right: 20, top: 16, bottom: 86 };
const PLOT_H = 300;
const HEIGHT = PAD.top + PLOT_H + PAD.bottom;

const COLORS: Record<Series, string> = { v: "#d1495b", a: "#2e86ab" };
const LABELS: Record<Series, string> = { v: "Valence", a: "Arousal" };

const clamp = (x: number) => Math.max(-1, Math.min(1, x));
const yOf = (val: number) => PAD.top + ((1 - val) / 2) * PLOT_H;
const valOf = (y: number) => clamp(1 - ((y - PAD.top) / PLOT_H) * 2);
const xOf = (i: number) => PAD.left + i * STEP_X + STEP_X / 2;

function polyline(rows: VaRow[], key: Series): string {
  return rows.map((r, i) => `${xOf(i)},${yOf(r[key])}`).join(" ");
}

function faceWeightsAt(rows: VaRow[], idx: number, smooth: boolean) {
  if (rows.length === 0) return fkcnToGrimace([1, 0, 0, 0, 0, 0, 0]);
  if (!smooth) return fkcnToGrimace(fkcnWeights(rows[idx].v, rows[idx].a));
  let ema = fkcnWeights(rows[0].v, rows[0].a);
  for (let i = 1; i <= idx; i += 1) {
    ema = emaBlend(ema, fkcnWeights(rows[i].v, rows[i].a), EMA_ALPHA);
  }
  return fkcnToGrimace(ema);
}

export function VaEditor() {
  const [rows, setRows] = useState<VaRow[]>([]);
  const [original, setOriginal] = useState<VaRow[]>([]);
  const [source, setSource] = useState("");
  const [status, setStatus] = useState("加载中…");
  const [dirty, setDirty] = useState(false);
  const [sel, setSel] = useState(0);
  const [smooth, setSmooth] = useState(true);
  const [showOriginal, setShowOriginal] = useState(true);
  const [playing, setPlaying] = useState(false);
  const [drag, setDrag] = useState<{ idx: number; key: Series } | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const load = useCallback(async (wantOriginal: boolean) => {
    const q = new URLSearchParams({ cond: COND, trial: TRIAL });
    if (wantOriginal) q.set("original", "1");
    const res = await fetch(`/__va/load?${q}`);
    const body = await res.json();
    if (!res.ok) {
      setStatus(`加载失败：${body.error}`);
      return;
    }
    setRows(parseVaCsv(body.text));
    setOriginal(parseVaCsv(body.originalText));
    setSource(body.source);
    setDirty(wantOriginal);
    setStatus(`已加载 ${body.source}`);
  }, []);

  useEffect(() => {
    void load(false);
  }, [load]);

  useEffect(() => {
    if (!dirty) return undefined;
    const warn = (e: BeforeUnloadEvent) => e.preventDefault();
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [dirty]);

  useEffect(() => {
    if (!playing || rows.length === 0) return undefined;
    if (sel >= rows.length - 1) {
      setPlaying(false);
      return undefined;
    }
    const dt = Math.max(0.12, rows[sel + 1].t - rows[sel].t);
    const id = window.setTimeout(() => setSel((s) => s + 1), dt * 1000);
    return () => window.clearTimeout(id);
  }, [playing, sel, rows]);

  const face = useMemo(() => faceWeightsAt(rows, sel, smooth), [rows, sel, smooth]);

  const setValue = useCallback((idx: number, key: Series, val: number) => {
    setRows((prev) => prev.map((r, i) => (i === idx ? { ...r, [key]: clamp(val) } : r)));
    setDirty(true);
  }, []);

  const svgY = (e: React.PointerEvent) => {
    const svg = svgRef.current;
    const ctm = svg?.getScreenCTM();
    if (!svg || !ctm) return 0;
    const pt = new DOMPoint(e.clientX, e.clientY).matrixTransform(ctm.inverse());
    return pt.y;
  };

  const onPointDown = (e: React.PointerEvent, idx: number, key: Series) => {
    e.stopPropagation();
    (e.target as Element).setPointerCapture(e.pointerId);
    setPlaying(false);
    setSel(idx);
    setDrag({ idx, key });
  };

  const onPointMove = (e: React.PointerEvent) => {
    if (!drag) return;
    setValue(drag.idx, drag.key, Number(valOf(svgY(e)).toFixed(4)));
  };

  const resetPoint = (idx: number) => {
    const o = original[idx];
    if (!o) return;
    setRows((prev) => prev.map((r, i) => (i === idx ? { ...r, v: o.v, a: o.a } : r)));
    setDirty(true);
  };

  const save = async () => {
    const q = new URLSearchParams({ cond: COND, trial: TRIAL });
    const res = await fetch(`/__va/save?${q}`, { method: "POST", body: toVaCsv(rows) });
    const body = await res.json();
    if (!res.ok) {
      setStatus(`保存失败：${body.error}`);
      return;
    }
    setDirty(false);
    setSource(body.saved);
    setStatus(`已保存到 ${body.saved}`);
  };

  const isEdited = (i: number, key: Series) =>
    original[i] !== undefined && Math.abs(original[i][key] - rows[i][key]) > 1e-6;
  const editedCount = rows.filter((_, i) => isEdited(i, "v") || isEdited(i, "a")).length;

  const width = PAD.left + rows.length * STEP_X + PAD.right;
  const cur = rows[sel];

  return (
    <div className="va-editor">
      <header className="va-head">
        <h1>
          VA 轨迹编辑 <span className="muted">{`${COND}/${TRIAL}`}</span>
        </h1>
        <div className="va-actions">
          <button type="button" onClick={() => setPlaying((p) => !p)}>
            {playing ? "暂停" : "播放"}
          </button>
          <label>
            <input type="checkbox" checked={smooth} onChange={(e) => setSmooth(e.target.checked)} />
            EMA 平滑（α={EMA_ALPHA}）
          </label>
          <label>
            <input
              type="checkbox"
              checked={showOriginal}
              onChange={(e) => setShowOriginal(e.target.checked)}
            />
            显示原始轨迹
          </label>
          <button type="button" onClick={() => void load(true)}>
            全部还原
          </button>
          <button type="button" className="primary" disabled={!dirty} onClick={() => void save()}>
            保存{dirty ? " *" : ""}
          </button>
        </div>
      </header>
      <p className="va-status muted">
        {status}
        {source ? ` · 已修改 ${editedCount} 个 token` : ""}
        {" · 上下拖动圆点改值，双击圆点还原该 token"}
      </p>

      <div className="va-main">
        <div className="va-chart">
          <svg
            ref={svgRef}
            width={width}
            height={HEIGHT}
            onPointerMove={onPointMove}
            onPointerUp={() => setDrag(null)}
            onPointerCancel={() => setDrag(null)}
          >
            {[-1, -0.5, 0, 0.5, 1].map((g) => (
              <g key={g}>
                <line
                  className={g === 0 ? "grid zero" : "grid"}
                  x1={PAD.left}
                  x2={width - PAD.right}
                  y1={yOf(g)}
                  y2={yOf(g)}
                />
                <text className="axis" x={PAD.left - 8} y={yOf(g) + 4} textAnchor="end">
                  {g}
                </text>
              </g>
            ))}

            {cur ? (
              <rect
                className="sel-band"
                x={xOf(sel) - STEP_X / 2}
                y={PAD.top}
                width={STEP_X}
                height={PLOT_H}
              />
            ) : null}

            {showOriginal
              ? (["v", "a"] as Series[]).map((k) => (
                  <polyline
                    key={`o-${k}`}
                    className="line original"
                    points={polyline(original, k)}
                    stroke={COLORS[k]}
                  />
                ))
              : null}

            {(["v", "a"] as Series[]).map((k) => (
              <polyline key={k} className="line" points={polyline(rows, k)} stroke={COLORS[k]} />
            ))}

            {(["v", "a"] as Series[]).map((k) =>
              rows.map((r, i) => (
                <circle
                  key={`${k}-${i}`}
                  className={`pt${isEdited(i, k) ? " edited" : ""}${i === sel ? " sel" : ""}`}
                  cx={xOf(i)}
                  cy={yOf(r[k])}
                  r={i === sel ? 6 : 4.5}
                  fill={COLORS[k]}
                  onPointerDown={(e) => onPointDown(e, i, k)}
                  onDoubleClick={() => resetPoint(i)}
                >
                  <title>{`${JSON.stringify(r.token)} t=${r.t.toFixed(2)}s ${k.toUpperCase()}=${r[k].toFixed(3)}`}</title>
                </circle>
              )),
            )}

            {rows.map((r, i) => (
              <text
                key={`tok-${i}`}
                className={`tok-label${i === sel ? " sel" : ""}`}
                transform={`translate(${xOf(i) + 4}, ${PAD.top + PLOT_H + 10}) rotate(-60)`}
                textAnchor="end"
                onClick={() => setSel(i)}
              >
                {r.token.trim() || "␣"}
              </text>
            ))}
          </svg>
        </div>

        <aside className="va-side">
          <div className="va-face">
            <GrimaceFace weights={face} />
          </div>
          <div className="va-legend">
            {(["v", "a"] as Series[]).map((k) => (
              <span key={k}>
                <i style={{ background: COLORS[k] }} />
                {LABELS[k]}
              </span>
            ))}
          </div>
          {cur ? (
            <div className="va-inspector">
              <div className="tok-name">
                #{sel} <b>{JSON.stringify(cur.token)}</b> <span className="muted">t={cur.t.toFixed(2)}s</span>
              </div>
              {(["v", "a"] as Series[]).map((k) => (
                <label key={k} className="va-field">
                  <span style={{ color: COLORS[k] }}>{LABELS[k]}</span>
                  <input
                    type="range"
                    min={-1}
                    max={1}
                    step={0.01}
                    value={cur[k]}
                    onChange={(e) => setValue(sel, k, Number(e.target.value))}
                  />
                  <input
                    type="number"
                    min={-1}
                    max={1}
                    step={0.01}
                    value={Number(cur[k].toFixed(3))}
                    onChange={(e) => setValue(sel, k, Number(e.target.value))}
                  />
                </label>
              ))}
              <div className="va-nav">
                <button type="button" disabled={sel <= 0} onClick={() => setSel((s) => s - 1)}>
                  ← 上一个
                </button>
                <button type="button" onClick={() => resetPoint(sel)}>
                  还原此点
                </button>
                <button
                  type="button"
                  disabled={sel >= rows.length - 1}
                  onClick={() => setSel((s) => s + 1)}
                >
                  下一个 →
                </button>
              </div>
            </div>
          ) : null}
        </aside>
      </div>
    </div>
  );
}
