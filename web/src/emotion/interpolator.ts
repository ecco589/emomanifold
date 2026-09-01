function easeInOut(t: number): number {
  const x = Math.max(0, Math.min(1, t));
  return x * x * (3 - 2 * x);
}

function lerp(a: number[], b: number[], t: number): number[] {
  return a.map((ai, i) => ai + ((b[i] ?? 0) - ai) * t);
}

function rateLimit(prev: number[], next: number[], maxDelta: number): number[] {
  return prev.map((p, i) => {
    const n = next[i] ?? 0;
    const d = Math.max(-maxDelta, Math.min(maxDelta, n - p));
    return p + d;
  });
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

/** Match desktop demo: 10fps, ease-in-out, per-frame delta cap. */
export class FaceAnimator {
  current: number[];
  target: number[];

  constructor(
    private fps = 10,
    private transitionMs = 200,
    private maxDelta = 0.05,
    private maxSettleFrames = 6,
    nDims = 6,
  ) {
    this.current = new Array(nDims).fill(0);
    this.target = new Array(nDims).fill(0);
  }

  getWeights(): number[] {
    return this.current.slice();
  }

  setTarget(weights: number[]): void {
    this.target = weights.slice();
  }

  snapToTarget(): void {
    this.current = this.target.slice();
  }

  /** Idle tick for rAF; returns null when unchanged. */
  step(): number[] | null {
    return this.current.slice();
  }

  async animateTo(target: number[]): Promise<void> {
    const tgt = target.slice();
    this.target = tgt;
    const start = this.current.slice();
    const frames = Math.max(2, Math.round(this.transitionMs / (1000 / this.fps)));
    const frameDt = 1000 / this.fps;

    for (let i = 1; i <= frames; i += 1) {
      const t = easeInOut(i / frames);
      const raw = lerp(start, tgt, t);
      this.current = rateLimit(this.current, raw, this.maxDelta);
      await sleep(frameDt);
    }

    for (let j = 0; j < this.maxSettleFrames; j += 1) {
      const settled = this.current.every(
        (c, i) => Math.abs(c - (tgt[i] ?? 0)) <= this.maxDelta * 0.5,
      );
      if (settled) break;
      this.current = rateLimit(this.current, tgt, this.maxDelta);
      await sleep(frameDt);
    }
  }
}
