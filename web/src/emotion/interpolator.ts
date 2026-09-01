export class FaceAnimator {
  current: number[];
  target: number[];

  constructor(
    private smoothFactor = 0.42,
    nDims = 6,
  ) {
    this.current = new Array(nDims).fill(0);
    this.target = new Array(nDims).fill(0);
  }

  setTarget(weights: number[]): void {
    this.target = weights.slice();
  }

  snapToTarget(): void {
    this.current = this.target.slice();
  }

  step(): number[] | null {
    let changed = false;
    this.current = this.current.map((c, i) => {
      const t = this.target[i] ?? 0;
      const next = c + (t - c) * this.smoothFactor;
      if (Math.abs(next - c) > 1e-4) changed = true;
      return next;
    });
    return changed ? this.current.slice() : null;
  }

  waitUntilSettled(opts?: { threshold?: number; timeoutMs?: number }): Promise<void> {
    const threshold = opts?.threshold ?? 0.05;
    const timeoutMs = opts?.timeoutMs ?? 130;
    const start = performance.now();
    return new Promise((resolve) => {
      const check = () => {
        const settled = this.current.every(
          (c, i) => Math.abs(c - (this.target[i] ?? 0)) < threshold,
        );
        if (settled || performance.now() - start >= timeoutMs) {
          resolve();
          return;
        }
        requestAnimationFrame(check);
      };
      check();
    });
  }
}
