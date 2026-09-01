export type VAD = {
  valence: number;
  arousal: number;
  dominance: number;
};

export type ScalerDim = { scale: number; bias: number };

export type PipelineConfig = {
  hidden_size: number;
  axes: {
    valence: number[];
    arousal: number[];
    dominance: number[];
  };
  scaler: {
    valence: ScalerDim;
    arousal: ScalerDim;
    dominance: ScalerDim;
  };
};

function l2Normalize(values: ArrayLike<number>): Float32Array {
  let sumSq = 0;
  for (let i = 0; i < values.length; i += 1) {
    sumSq += values[i] * values[i];
  }
  const denom = Math.sqrt(sumSq) + 1e-8;
  const out = new Float32Array(values.length);
  for (let i = 0; i < values.length; i += 1) {
    out[i] = values[i] / denom;
  }
  return out;
}

function dot(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const n = Math.min(a.length, b.length);
  let s = 0;
  for (let i = 0; i < n; i += 1) {
    s += a[i] * b[i];
  }
  return s;
}

function applyScaler(raw: number, dim: ScalerDim): number {
  return Math.max(-1, Math.min(1, dim.scale * raw + dim.bias));
}

export function projectHidden(hidden: ArrayLike<number>, pipeline: PipelineConfig): VAD {
  const h = l2Normalize(hidden);
  return {
    valence: applyScaler(dot(h, pipeline.axes.valence), pipeline.scaler.valence),
    arousal: applyScaler(dot(h, pipeline.axes.arousal), pipeline.scaler.arousal),
    dominance: applyScaler(dot(h, pipeline.axes.dominance), pipeline.scaler.dominance),
  };
}
