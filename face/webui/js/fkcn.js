/**
 * FKCN: (V, A) → 7-dim basic emotion weights
 * Mirrors src/fkcn.py
 */
export const EMOTION_LABELS = [
  "Neutral",
  "Happiness",
  "Surprise",
  "Fear",
  "Sadness",
  "Disgust",
  "Anger",
];

function oneHot(index, size = 7) {
  const v = new Array(size).fill(0);
  v[index] = 1;
  return v;
}

const DEFAULT_PROTOTYPES = [
  [[0.61, -0.47], oneHot(0)],
  [[0.81, 0.66], oneHot(1)],
  [[0.88, 0.59], oneHot(1)],
  [[-0.75, 0.61], oneHot(2)],
  [[-0.71, 0.58], oneHot(2)],
  [[-0.83, 0.54], oneHot(3)],
  [[-0.91, 0.52], oneHot(3)],
  [[-0.47, -0.56], oneHot(4)],
  [[-0.54, -0.59], oneHot(4)],
  [[-0.95, -0.06], oneHot(5)],
  [[-0.84, -0.19], oneHot(5)],
  [[-0.95, 0.17], oneHot(6)],
  [[-0.98, 0.1], oneHot(6)],
];

/** Label anchor positions (centroids) for the 7 emotions on VA plane */
export const EMOTION_ANCHORS = [
  { label: "Neutral", v: 0.61, a: -0.47 },
  { label: "Happiness", v: 0.845, a: 0.625 },
  { label: "Surprise", v: -0.73, a: 0.595 },
  { label: "Fear", v: -0.87, a: 0.53 },
  { label: "Sadness", v: -0.505, a: -0.575 },
  { label: "Disgust", v: -0.895, a: -0.125 },
  { label: "Anger", v: -0.965, a: 0.135 },
];

function squaredDistances(v, a, prototypes) {
  return prototypes.map(([coord]) => {
    const dv = v - coord[0];
    const da = a - coord[1];
    return dv * dv + da * da;
  });
}

function membership(distances) {
  const eps = 1e-12;
  const n = distances.length;
  const mem = new Array(n).fill(0);

  const zeroIdx = distances
    .map((d, i) => (d <= eps ? i : -1))
    .filter((i) => i >= 0);

  if (zeroIdx.length > 0) {
    const share = 1 / zeroIdx.length;
    zeroIdx.forEach((i) => {
      mem[i] = share;
    });
    return mem;
  }

  let invSum = 0;
  const inv = distances.map((d) => 1 / (d + eps));
  invSum = inv.reduce((s, x) => s + x, 0);
  for (let i = 0; i < n; i += 1) {
    const denom = distances[i] * invSum;
    mem[i] = denom > 0 ? 1 / denom : 0;
  }
  return mem;
}

export function computeFKCN(v, a, prototypes = DEFAULT_PROTOTYPES) {
  const dist = squaredDistances(v, a, prototypes);
  const mem = membership(dist);
  const weights = new Array(7).fill(0);
  prototypes.forEach(([, wVec], i) => {
    const m = mem[i];
    for (let j = 0; j < 7; j += 1) {
      weights[j] += m * wVec[j];
    }
  });
  return weights;
}

export function fkcnToGrimace(weights7) {
  return weights7.slice(1, 7);
}

export function clampVA(v, a) {
  return [
    Math.max(-1, Math.min(1, v)),
    Math.max(-1, Math.min(1, a)),
  ];
}

/** Emotional intensity for heatmap alpha: 1 − neutral share */
export function emotionIntensity(weights7) {
  return Math.max(0, Math.min(1, 1 - (weights7[0] || 0)));
}

export { DEFAULT_PROTOTYPES };
