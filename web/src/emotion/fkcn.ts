/** Port of src/fkcn.py DEFAULT_PROTOTYPES + membership. */

const LABELS = [
  "neutral",
  "happiness",
  "surprise",
  "fear",
  "sadness",
  "disgust",
  "anger",
] as const;

function oneHot(index: number): number[] {
  const vec = new Array(LABELS.length).fill(0);
  vec[index] = 1;
  return vec;
}

const PROTOTYPES: { xy: [number, number]; w: number[] }[] = [
  { xy: [0.61, -0.47], w: oneHot(0) },
  { xy: [0.81, 0.66], w: oneHot(1) },
  { xy: [0.88, 0.59], w: oneHot(1) },
  { xy: [-0.75, 0.61], w: oneHot(2) },
  { xy: [-0.71, 0.58], w: oneHot(2) },
  { xy: [-0.83, 0.54], w: oneHot(3) },
  { xy: [-0.91, 0.52], w: oneHot(3) },
  { xy: [-0.47, -0.56], w: oneHot(4) },
  { xy: [-0.54, -0.59], w: oneHot(4) },
  { xy: [-0.95, -0.06], w: oneHot(5) },
  { xy: [-0.84, -0.19], w: oneHot(5) },
  { xy: [-0.95, 0.17], w: oneHot(6) },
  { xy: [-0.98, 0.1], w: oneHot(6) },
];

const EPS = 1e-12;

export function fkcnWeights(valence: number, arousal: number): number[] {
  const d2 = PROTOTYPES.map(({ xy }) => {
    const dx = valence - xy[0];
    const dy = arousal - xy[1];
    return dx * dx + dy * dy;
  });

  const membership = new Array(d2.length).fill(0);
  const zeroIdx = d2.map((d, i) => (d <= EPS ? i : -1)).filter((i) => i >= 0);

  if (zeroIdx.length > 0) {
    const share = 1 / zeroIdx.length;
    for (const i of zeroIdx) membership[i] = share;
  } else {
    let invSum = 0;
    for (const d of d2) {
      invSum += 1 / (d + EPS);
    }
    for (let i = 0; i < d2.length; i += 1) {
      membership[i] = 1 / (d2[i] * invSum);
    }
  }

  const out = new Array(LABELS.length).fill(0);
  for (let i = 0; i < PROTOTYPES.length; i += 1) {
    const w = PROTOTYPES[i].w;
    for (let k = 0; k < out.length; k += 1) {
      out[k] += membership[i] * w[k];
    }
  }
  return out;
}

export function fkcnToGrimace(fkcn: number[]): {
  joy: number;
  surprise: number;
  fear: number;
  sadness: number;
  disgust: number;
  anger: number;
} {
  return {
    joy: fkcn[1] ?? 0,
    surprise: fkcn[2] ?? 0,
    fear: fkcn[3] ?? 0,
    sadness: fkcn[4] ?? 0,
    disgust: fkcn[5] ?? 0,
    anger: fkcn[6] ?? 0,
  };
}

export function emaBlend(prev: number[], cur: number[], alpha = 0.6): number[] {
  return cur.map((c, i) => alpha * c + (1 - alpha) * (prev[i] ?? 0));
}
