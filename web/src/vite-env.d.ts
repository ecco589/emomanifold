/// <reference types="vite/client" />
/// <reference path="../../grimace-main/src/facedata.d.ts" />

interface ImportMetaEnv {
  readonly VITE_API_URL?: string;
  readonly VITE_BASE?: string;
}

declare module "grimace" {
  export type EmotionSet = {
    anger?: number;
    disgust?: number;
    fear?: number;
    joy?: number;
    sadness?: number;
    surprise?: number;
    [emotionId: string]: number | undefined;
  };

  export default class Grimace {
    constructor(container: HTMLElement);
    setEmotionSet(emotionSet: EmotionSet, duration?: number): void;
    setEmotion(emotion: string, value: number, duration?: number): void;
  }
}
