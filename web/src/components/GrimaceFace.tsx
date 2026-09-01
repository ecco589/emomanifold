import { useEffect, useRef } from "react";
import Grimace from "grimace";

type Weights = {
  joy: number;
  surprise: number;
  fear: number;
  sadness: number;
  disgust: number;
  anger: number;
};

export function GrimaceFace({ weights }: { weights: Weights }) {
  const hostRef = useRef<HTMLDivElement>(null);
  const faceRef = useRef<Grimace | null>(null);
  const weightsRef = useRef(weights);
  weightsRef.current = weights;

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return undefined;

    let cancelled = false;
    let attempts = 0;

    const ensureFace = () => {
      if (cancelled || faceRef.current) return;
      const { width, height } = host.getBoundingClientRect();
      if (width < 8 || height < 8) return;
      try {
        faceRef.current = new Grimace(host);
        faceRef.current.setEmotionSet(weightsRef.current, 0);
      } catch (err) {
        console.error("Grimace init failed:", err);
      }
    };

    const retry = () => {
      ensureFace();
      if (!faceRef.current && !cancelled && attempts < 60) {
        attempts += 1;
        requestAnimationFrame(retry);
      }
    };

    retry();
    const ro = new ResizeObserver(() => ensureFace());
    ro.observe(host);

    return () => {
      cancelled = true;
      ro.disconnect();
      host.replaceChildren();
      faceRef.current = null;
    };
  }, []);

  useEffect(() => {
    faceRef.current?.setEmotionSet(weights, 0);
  }, [weights]);

  return <div className="face-canvas" ref={hostRef} />;
}
