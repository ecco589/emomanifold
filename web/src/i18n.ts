export type Lang = "en" | "zh";

export const copy = {
  en: {
    brand: "EmoManifold",
    paperTitle:
      "“Written All Over the Face”: Driving Facial Robots via LLM Internal Emotion Manifolds",
    authors: "",
    links: {
      github: "GitHub",
      arxiv: "arXiv",
    },
    langSwitch: "中文",
    abstractTitle: "Abstract",
    abstract:
      "The face is the most natural channel for conveying emotion in human–robot interaction. Existing LLM-driven pipelines rely on sentence-level discrete emotion labels, so updates lag behind speech and miss fine-grained intra-utterance shifts. EmoManifold reads a continuous affective signal from hidden states during streaming decoding—without asking the model to verbalize an emotion label—and drives facial expression in real time, keeping the face synchronized with token generation.",
    notice:
      "Runs in your browser — no model download. Text is generated on our server; the face is driven on your device.",
    noticeLocal: "Facial animation is rendered on your device.",
    demoTitle: "Live demo",
    demoHint: "Hover a token to see valence, arousal, and dominance.",
    placeholder: "Type a message and press Enter",
    placeholderLoading: "…",
    send: "Send",
    stop: "Stop",
    examplesLabel: "Try",
    examples: [
      "I just got the job — I still cannot believe it.",
      "I am sorry. I know this must be a terrible day.",
      "Oh great, another delay. That is exactly what I needed.",
    ],
    methodTitle: "How it works",
    methodCaption:
      "Online decoding extracts the last-layer hidden state as each token is produced, projects it onto VAD, and maps valence–arousal through FKCN to a Grimace face. Axes are built once, offline, from pole words and the unembedding matrix.",
    loadConnecting: "Connecting to server…",
    loadNeedGpu:
      "WebGPU is not available. Please open this page in desktop Chrome or Edge.",
    loadDownloading: "Downloading the on-device model",
    loadPhaseTokenizer: "Step 1/2 · tokenizer",
    loadPhaseOnnx: "Step 2/2 · ONNX weights",
    loadInitializing: "Compiling on-device runtime…",
    loadReady: "Model ready. Your device will run every token.",
    loadError: "Failed to load the model.",
    loadStuck: "Load stalled — no response.",
    loadRetry: "Retry",
    hiddenMissing:
      "Tokens are streaming, but this ONNX build does not expose hidden states yet, so the face is not driven by the paper readout. Next step: export an ONNX graph that outputs last-layer hₜ.",
    emptyChat: "The face will move as the model writes.",
    you: "You",
    model: "EmoManifold",
    footer:
      "Project page MVP. Code and arXiv links will be filled in after release.",
  },
  zh: {
    brand: "EmoManifold",
    paperTitle:
      "“Written All Over the Face”: Driving Facial Robots via LLM Internal Emotion Manifolds",
    authors: "",
    links: {
      github: "GitHub",
      arxiv: "arXiv",
    },
    langSwitch: "EN",
    abstractTitle: "摘要",
    abstract:
      "面部是人机交互中最自然的情绪通道。现有大模型方案多依赖句级离散情绪标签，表情更新滞后，也难以刻画句内转折。EmoManifold 不令模型“说出”情绪，而是在流式解码中从隐状态读取连续情感信号并实时驱动表情，使面部与 token 生成同步。",
    notice:
      "在浏览器中运行，无需下载模型。文本由服务器生成，表情在您的设备上驱动。",
    noticeLocal: "面部表情在您的设备上渲染。",
    demoTitle: "在线演示",
    demoHint: "将鼠标悬停在 token 上可查看效价、唤醒度与支配度。",
    placeholder: "输入消息，按回车发送",
    placeholderLoading: "…",
    send: "发送",
    stop: "停止",
    examplesLabel: "试一试",
    examples: [
      "我拿到 offer 了，到现在都不敢相信。",
      "真的很抱歉。我知道这对你来说会是很难的一天。",
      "太好了，又延误了。我正需要这样。",
    ],
    methodTitle: "方法",
    methodCaption:
      "在线解码在每个 token 产生时提取末层隐状态，投影到 VAD，再经 FKCN 映射为 Grimace 表情。情感轴由极词与解嵌入矩阵离线一次性构造。",
    loadConnecting: "正在连接服务器…",
    loadNeedGpu: "当前浏览器没有 WebGPU。请使用桌面版 Chrome 或 Edge 打开。",
    loadDownloading: "正在下载端侧模型",
    loadPhaseTokenizer: "第 1/2 步 · tokenizer",
    loadPhaseOnnx: "第 2/2 步 · ONNX 权重",
    loadInitializing: "正在编译端侧运行时…",
    loadReady: "模型已就绪。每个 token 都在您的设备上计算。",
    loadError: "模型加载失败。",
    loadStuck: "加载卡住 — 长时间无响应。",
    loadRetry: "重试",
    hiddenMissing:
      "文本已在流式输出，但当前 ONNX 尚未暴露隐状态，表情还不是论文中的读回。下一步：导出带末层 hₜ 的 ONNX。",
    emptyChat: "模型开始写字时，表情会跟着动。",
    you: "你",
    model: "EmoManifold",
    footer: "项目主页 MVP。代码与 arXiv 链接将在公开后补全。",
  },
} as const;
