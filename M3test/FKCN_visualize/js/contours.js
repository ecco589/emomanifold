/** Marching squares — iso-weight contours on VA grid */

export function sampleWeightGrid(gridRes, emotionIdx, computeFKCN) {
  const n = gridRes;
  const values = new Float32Array(n * n);
  for (let iy = 0; iy < n; iy += 1) {
    for (let ix = 0; ix < n; ix += 1) {
      const v = (ix / (n - 1)) * 2 - 1;
      const a = 1 - (iy / (n - 1)) * 2;
      values[iy * n + ix] = computeFKCN(v, a)[emotionIdx];
    }
  }
  return { n, values };
}

function interpEdge(v1, v2, level) {
  if (Math.abs(v2 - v1) < 1e-9) return 0.5;
  return (level - v1) / (v2 - v1);
}

/**
 * Standard marching squares. Edge indices: 0=bottom, 1=right, 2=top, 3=left.
 * Returns segments as [[x0,y0],[x1,y1]] in normalised grid coords.
 */
export function marchingSquares(values, n, level) {
  const segments = [];

  const val = (ix, iy) => values[iy * n + ix];

  const cellX = (ix) => ix / (n - 1);
  const cellY = (iy) => iy / (n - 1);
  const d = 1 / (n - 1);

  const edgePoint = (ix, iy, e, v0, v1, v2, v3) => {
    const x0 = cellX(ix);
    const y0 = cellY(iy);
    switch (e) {
      case 0:
        return [x0 + d * interpEdge(v0, v1, level), y0];
      case 1:
        return [x0 + d, y0 + d * interpEdge(v1, v2, level)];
      case 2:
        return [x0 + d * interpEdge(v3, v2, level), y0 + d];
      case 3:
        return [x0, y0 + d * interpEdge(v0, v3, level)];
      default:
        return [x0, y0];
    }
  };

  /** [eA, eB] pairs per case (Paul Bourke LUT, simplified) */
  const LUT = [
    [], [0, 3], [0, 1], [1, 3], [1, 2], [0, 3, 1, 2], [0, 2], [2, 3],
    [2, 3], [0, 2], [0, 3, 1, 2], [1, 2], [1, 3], [0, 1], [0, 3], [],
  ];

  for (let iy = 0; iy < n - 1; iy += 1) {
    for (let ix = 0; ix < n - 1; ix += 1) {
      const v0 = val(ix, iy);
      const v1 = val(ix + 1, iy);
      const v2 = val(ix + 1, iy + 1);
      const v3 = val(ix, iy + 1);

      let idx = 0;
      if (v0 >= level) idx |= 1;
      if (v1 >= level) idx |= 2;
      if (v2 >= level) idx |= 4;
      if (v3 >= level) idx |= 8;

      const edges = LUT[idx];
      for (let i = 0; i < edges.length; i += 2) {
        const a = edgePoint(ix, iy, edges[i], v0, v1, v2, v3);
        const b = edgePoint(ix, iy, edges[i + 1], v0, v1, v2, v3);
        segments.push([a, b]);
      }
    }
  }
  return segments;
}

/** Single-emotion detail: 3 nested iso-lines */
export const DETAIL_LEVELS = [0.35, 0.55, 0.75];

export const DETAIL_STYLE = {
  0.35: { alpha: 0.28, width: 0.6, dash: [3, 3] },
  0.55: { alpha: 0.48, width: 0.8, dash: [] },
  0.75: { alpha: 0.72, width: 1.0, dash: [] },
};

/** Overview: one iso-line per emotion */
export const OVERVIEW_LEVEL = 0.55;

export const OVERVIEW_STYLE = { alpha: 0.38, width: 0.7, dash: [] };
