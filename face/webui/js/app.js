import {
  EMOTION_LABELS,
  EMOTION_ANCHORS,
  DEFAULT_PROTOTYPES,
  computeFKCN,
  fkcnToGrimace,
  clampVA,
} from "./fkcn.js";
import {
  sampleWeightGrid,
  marchingSquares,
  DETAIL_LEVELS,
  DETAIL_STYLE,
  OVERVIEW_LEVEL,
  OVERVIEW_STYLE,
} from "./contours.js";

const state = {
  v: 0.2,
  a: 0.1,
  showContours: true,
  showInternalProtos: false,
  showAnchors: true,
  showLabels: true,
  contourTarget: "1",
  gridRes: 100,
};

const canvas = document.getElementById("va-canvas");
const ctx = canvas.getContext("2d");
const inputV = document.getElementById("input-v");
const inputA = document.getElementById("input-a");
const toggleContours = document.getElementById("toggle-contours");
const toggleInternal = document.getElementById("toggle-internal");
const toggleAnchors = document.getElementById("toggle-anchors");
const toggleLabels = document.getElementById("toggle-labels");
const contourEmotion = document.getElementById("contour-emotion");
const weightsRoot = document.getElementById("weights-list");

let grimaceReady = false;
let contourCache = {};
const PAD = { l: 52, r: 18, t: 18, b: 46 };

function cacheKey(emotionIdx, level) {
  return `${emotionIdx}:${level}`;
}

function rebuildContourCache() {
  contourCache = {};
  const store = (emotionIdx, level) => {
    const { n, values } = sampleWeightGrid(state.gridRes, emotionIdx, computeFKCN);
    contourCache[cacheKey(emotionIdx, level)] = marchingSquares(values, n, level);
  };

  DETAIL_LEVELS.forEach((level) => {
    for (let k = 0; k < 7; k += 1) store(k, level);
  });
  for (let k = 0; k < 7; k += 1) store(k, OVERVIEW_LEVEL);
}

function resizeCanvas() {
  const wrap = canvas.parentElement;
  const dpr = window.devicePixelRatio || 1;
  const w = wrap.clientWidth;
  const h = wrap.clientHeight;
  canvas.width = Math.floor(w * dpr);
  canvas.height = Math.floor(h * dpr);
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}

function plotRect() {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  return {
    x: PAD.l,
    y: PAD.t,
    w: w - PAD.l - PAD.r,
    h: h - PAD.t - PAD.b,
  };
}

function vaToXY(v, a, rect) {
  return {
    x: rect.x + ((v + 1) / 2) * rect.w,
    y: rect.y + ((1 - a) / 2) * rect.h,
  };
}

function gridToXY(gx, gy, rect) {
  const v = gx * 2 - 1;
  const a = 1 - gy * 2;
  return vaToXY(v, a, rect);
}

function xyToVA(x, y, rect) {
  const v = ((x - rect.x) / rect.w) * 2 - 1;
  const a = 1 - ((y - rect.y) / rect.h) * 2;
  return clampVA(v, a);
}

function drawSegments(segments, rect, style) {
  ctx.save();
  ctx.beginPath();
  ctx.rect(rect.x, rect.y, rect.w, rect.h);
  ctx.clip();
  ctx.strokeStyle = `rgba(185, 28, 28, ${style.alpha})`;
  ctx.lineWidth = style.width;
  ctx.setLineDash(style.dash || []);
  ctx.beginPath();
  segments.forEach(([a, b]) => {
    const p0 = gridToXY(a[0], a[1], rect);
    const p1 = gridToXY(b[0], b[1], rect);
    ctx.moveTo(p0.x, p0.y);
    ctx.lineTo(p1.x, p1.y);
  });
  ctx.stroke();
  ctx.restore();
}

function drawIsoContours(rect) {
  if (!Object.keys(contourCache).length) rebuildContourCache();

  if (state.contourTarget === "all") {
    for (let k = 0; k < 7; k += 1) {
      const segments = contourCache[cacheKey(k, OVERVIEW_LEVEL)] || [];
      drawSegments(segments, rect, { ...OVERVIEW_STYLE, alpha: 0.22 + k * 0.03 });
    }
    return;
  }

  const k = parseInt(state.contourTarget, 10);
  DETAIL_LEVELS.forEach((level) => {
    const segments = contourCache[cacheKey(k, level)] || [];
    drawSegments(segments, rect, DETAIL_STYLE[level]);
  });
}

function drawAxes(rect) {
  ctx.save();
  ctx.strokeStyle = "#1d1d1f";
  ctx.lineWidth = 0.75;
  ctx.fillStyle = "#1d1d1f";
  ctx.font = "11px -apple-system, Helvetica Neue, Arial, sans-serif";

  ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);

  ctx.strokeStyle = "#ececee";
  ctx.lineWidth = 0.5;
  for (let t = -1; t <= 1; t += 0.5) {
    if (t === 0) continue;
    const px = vaToXY(t, -1, rect);
    const py = vaToXY(-1, t, rect);
    ctx.beginPath();
    ctx.moveTo(px.x, rect.y);
    ctx.lineTo(px.x, rect.y + rect.h);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(rect.x, py.y);
    ctx.lineTo(rect.x + rect.w, py.y);
    ctx.stroke();
  }

  ctx.strokeStyle = "#86868b";
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  const zv = vaToXY(0, -1, rect);
  ctx.moveTo(zv.x, rect.y);
  ctx.lineTo(zv.x, rect.y + rect.h);
  const za = vaToXY(-1, 0, rect);
  ctx.moveTo(rect.x, za.y);
  ctx.lineTo(rect.x + rect.w, za.y);
  ctx.stroke();
  ctx.setLineDash([]);

  [-1, -0.5, 0, 0.5, 1].forEach((tick) => {
    const p = vaToXY(tick, -1, rect);
    ctx.fillText(tick.toFixed(1), p.x - 8, rect.y + rect.h + 16);
    const q = vaToXY(-1, tick, rect);
    ctx.fillText(tick.toFixed(1), rect.x - 28, q.y + 4);
  });

  ctx.fillText("Valence", rect.x + rect.w / 2 - 24, rect.y + rect.h + 34);
  ctx.save();
  ctx.translate(rect.x - 38, rect.y + rect.h / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Arousal", -22, 0);
  ctx.restore();
  ctx.restore();
}

/** 13 internal FKCN prototypes (optional, off by default) */
function drawInternalPrototypes(rect) {
  ctx.save();
  DEFAULT_PROTOTYPES.forEach(([coord]) => {
    const p = vaToXY(coord[0], coord[1], rect);
    ctx.beginPath();
    ctx.fillStyle = "rgba(110, 110, 115, 0.45)";
    ctx.arc(p.x, p.y, 2, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.restore();
}

/** 7 emotion anchor points — one per basic emotion */
function drawEmotionAnchors(rect) {
  ctx.save();
  EMOTION_ANCHORS.forEach(({ label, v, a }) => {
    const p = vaToXY(v, a, rect);
    ctx.beginPath();
    ctx.fillStyle = "#1d1d1f";
    ctx.arc(p.x, p.y, 3.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 1;
    ctx.arc(p.x, p.y, 3.5, 0, Math.PI * 2);
    ctx.stroke();
  });
  ctx.restore();
}

function drawLabels(rect) {
  ctx.save();
  ctx.font = "600 10px -apple-system, Helvetica Neue, Arial, sans-serif";
  EMOTION_ANCHORS.forEach(({ label, v, a }) => {
    const p = vaToXY(v, a, rect);
    ctx.fillStyle = "rgba(185, 28, 28, 0.85)";
    ctx.fillText(label, p.x + 6, p.y - 6);
  });
  ctx.restore();
}

function drawCurrentPoint(rect) {
  const p = vaToXY(state.v, state.a, rect);
  ctx.save();
  ctx.strokeStyle = "#1d1d1f";
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.arc(p.x, p.y, 7, 0, Math.PI * 2);
  ctx.stroke();
  ctx.fillStyle = "rgba(185, 28, 28, 0.18)";
  ctx.fill();
  ctx.beginPath();
  ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
  ctx.fillStyle = "#b91c1c";
  ctx.fill();
  ctx.font = "10px -apple-system, Helvetica Neue, Arial, sans-serif";
  ctx.fillStyle = "#6e6e73";
  ctx.fillText(`(${state.v.toFixed(2)}, ${state.a.toFixed(2)})`, p.x + 10, p.y + 14);
  ctx.restore();
}

function updateWeightsUI(weights7) {
  weightsRoot.innerHTML = "";
  weights7.forEach((w, i) => {
    const row = document.createElement("div");
    row.className = "weight-row";
    row.innerHTML = `
      <span>${EMOTION_LABELS[i]}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${Math.min(100, w * 100)}%"></div></div>
      <span class="weight-val">${w.toFixed(3)}</span>
    `;
    weightsRoot.appendChild(row);
  });
}

function updateGrimace(weights7) {
  if (!grimaceReady || typeof window.setWeights !== "function") return;
  window.setWeights(fkcnToGrimace(weights7));
}

function draw() {
  const rect = plotRect();
  ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  drawAxes(rect);
  if (state.showContours) drawIsoContours(rect);
  if (state.showInternalProtos) drawInternalPrototypes(rect);
  if (state.showAnchors) drawEmotionAnchors(rect);
  if (state.showLabels) drawLabels(rect);

  const weights7 = computeFKCN(state.v, state.a);
  drawCurrentPoint(rect);
  updateWeightsUI(weights7);
  updateGrimace(weights7);
}

// --- Robot mode: POST VA to backend, which drives the servos ---------------
const toggleRobot = document.getElementById("toggle-robot");
const robotStatus = document.getElementById("robot-status");
const robot = { on: false, timer: null, inflight: false, pending: false };

function setRobotStatus(text, cls) {
  robotStatus.textContent = text;
  robotStatus.className = "robot-status" + (cls ? " " + cls : "");
}

async function sendVA() {
  if (robot.inflight) {
    robot.pending = true;
    return;
  }
  robot.inflight = true;
  try {
    const res = await fetch("/apply", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ v: state.v, a: state.a, time: 300 }),
    });
    const data = await res.json();
    if (data.ok) {
      const mode = data.dry_run ? "dry-run" : "live";
      setRobotStatus(
        `${mode} · sent (${state.v.toFixed(2)}, ${state.a.toFixed(2)})`,
        data.dry_run ? "dry" : "live",
      );
    } else {
      setRobotStatus("error: " + (data.error || "unknown"), "err");
    }
  } catch (e) {
    setRobotStatus("backend offline", "err");
  } finally {
    robot.inflight = false;
    if (robot.pending) {
      robot.pending = false;
      sendVA();
    }
  }
}

function scheduleSend() {
  if (!robot.on) return;
  if (robot.timer) clearTimeout(robot.timer);
  robot.timer = setTimeout(sendVA, 120);
}

toggleRobot.addEventListener("change", () => {
  robot.on = toggleRobot.checked;
  if (robot.on) {
    setRobotStatus("connecting…");
    sendVA();
  } else {
    setRobotStatus("已停止");
  }
});

function setVA(v, a) {
  [state.v, state.a] = clampVA(v, a);
  inputV.value = state.v.toFixed(2);
  inputA.value = state.a.toFixed(2);
  draw();
  scheduleSend();
}

inputV.addEventListener("input", () => setVA(parseFloat(inputV.value) || 0, state.a));
inputA.addEventListener("input", () => setVA(state.v, parseFloat(inputA.value) || 0));

toggleContours.addEventListener("change", () => {
  state.showContours = toggleContours.checked;
  draw();
});
toggleInternal.addEventListener("change", () => {
  state.showInternalProtos = toggleInternal.checked;
  draw();
});
toggleAnchors.addEventListener("change", () => {
  state.showAnchors = toggleAnchors.checked;
  draw();
});
toggleLabels.addEventListener("change", () => {
  state.showLabels = toggleLabels.checked;
  draw();
});

contourEmotion.addEventListener("change", () => {
  state.contourTarget = contourEmotion.value;
  draw();
});

const settingsPanel = document.getElementById("settings-panel");
const settingsToggle = document.getElementById("settings-toggle");
settingsToggle.addEventListener("click", () => {
  const collapsed = settingsPanel.classList.toggle("collapsed");
  settingsToggle.setAttribute("aria-expanded", String(!collapsed));
});

canvas.addEventListener("click", (e) => {
  const rect = canvas.getBoundingClientRect();
  const plot = plotRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  if (x < plot.x || x > plot.x + plot.w || y < plot.y || y > plot.y + plot.h) return;
  const [v, a] = xyToVA(x, y, plot);
  setVA(v, a);
});

window.addEventListener("resize", () => {
  rebuildContourCache();
  resizeCanvas();
});

async function initGrimace() {
  window.module = { exports: {} };
  window.exports = window.module.exports;
  window.require = (name) => {
    if (name && name !== "ts-deepmerge") return {};
    const isObject = (v) => v && typeof v === "object" && !Array.isArray(v);
    const merge = (t, s) => {
      if (Array.isArray(t) && Array.isArray(s)) return t.concat(s);
      if (!isObject(t) || !isObject(s)) return s;
      const out = { ...t };
      Object.keys(s).forEach((k) => {
        out[k] = k in out ? merge(out[k], s[k]) : s[k];
      });
      return out;
    };
    return (...items) => items.reduce((acc, item) => merge(acc, item), {});
  };

  const script = document.createElement("script");
  script.src = "https://unpkg.com/grimace/dist/index.js";
  script.onload = () => {
    window.GrimaceModule = window.module.exports;
    const GrimaceCtor = window.GrimaceModule?.default || window.GrimaceModule;
    const container = document.getElementById("grimace-container");
    window.grimaceInstance = new GrimaceCtor(container);
    window.setWeights = (weights) => {
      const w = weights || [0, 0, 0, 0, 0, 0];
      const maxWeight = Math.max(...w, 0.01);
      const scale = maxWeight > 0 ? Math.min(1 / maxWeight, 2) : 1;
      const amplified = w.map((x) => Math.min(1, x * scale));
      window.grimaceInstance.setEmotionSet(
        {
          joy: amplified[0],
          surprise: amplified[1],
          fear: amplified[2],
          sadness: amplified[3],
          disgust: amplified[4],
          anger: amplified[5],
        },
        0,
      );
    };
    grimaceReady = true;
    draw();
  };
  document.body.appendChild(script);
}

rebuildContourCache();
initGrimace();
resizeCanvas();
setVA(state.v, state.a);
