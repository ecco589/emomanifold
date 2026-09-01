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

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return undefined;
    const start = () => {
      if (!faceRef.current && host.clientWidth > 0) {
        faceRef.current = new Grimace(host);
      }
    };
    start();
    const frame = requestAnimationFrame(start);
    const ro = new ResizeObserver(start);
    ro.observe(host);
    return () => {
      cancelAnimationFrame(frame);
      ro.disconnect();
    };
  }, []);

  useEffect(() => {
    faceRef.current?.setEmotionSet(weights, 0);
  }, [weights]);

  return <div className="face-canvas" ref={hostRef} />;
}
