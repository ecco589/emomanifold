export class FaceAnimator {
  current: number[];
  target: number[];
  private lastTs = 0;

  constructor(
    private fps = 10,
    private maxDelta = 0.08,
    nDims = 6,
  ) {
    this.current = new Array(nDims).fill(0);
    this.target = new Array(nDims).fill(0);
  }

  setTarget(weights: number[]): void {
    this.target = weights.slice();
  }

  step(now: number): number[] | null {
    const interval = 1000 / this.fps;
    if (now - this.lastTs < interval) return null;
    this.lastTs = now;
    this.current = this.current.map((c, i) => {
      const t = this.target[i] ?? 0;
      const d = Math.max(-this.maxDelta, Math.min(this.maxDelta, t - c));
      return c + d;
    });
    return this.current;
  }
}
