"""
用 NRC VAD 官方值更新 vad_lexicon_chinese.json。

优先匹配简体中文 NRC 词表（OneFilePerLanguage/Chinese-Simplified-NRC-VAD-Lexicon.txt）；
同一中文对应多条英文时取 V/A/D 均值。
未命中时回退到英文 NRC（NRC-VAD-Lexicon.txt），通过预置中英情感词对照表查词。

主词库存储 NRC 原生 0~1 区间；AdaptiveScaler 训练/推理时再映射到 [-1, 1]。
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import config

NRC_ZH_PATH = config.DATA_DIR / "NRC-VAD-Lexicon" / "OneFilePerLanguage" / "Chinese-Simplified-NRC-VAD-Lexicon.txt"
NRC_EN_PATH = config.DATA_DIR / "NRC-VAD-Lexicon" / "NRC-VAD-Lexicon.txt"
MASTER_LEXICON_PATH = config.VAD_LEXICON_DIR / "vad_lexicon_chinese.json"

# 简体中文 NRC 未覆盖时的英文回退（对应 NRC-VAD-Lexicon.txt 词条）
CHINESE_TO_ENGLISH_FALLBACK: Dict[str, str] = {
    "开心": "happy",
    "热爱": "love",
    "友好": "friendly",
    "甜蜜": "sweet",
    "美好": "beautiful",
    "骄傲": "pride",
    "安逸": "cozy",
    "哀伤": "mournful",
    "难过": "sad",
    "憎恨": "hatred",
    "厌烦": "annoyance",
    "嫌弃": "disdain",
    "凄惨": "wretched",
    "悲戚": "mournful",
    "失落": "dejected",
    "煎熬": "torment",
    "暴怒": "furious",
    "紧绷": "tense",
    "心慌": "nervous",
    "亢奋": "manic",
    "惊恐": "horrified",
    "担忧": "worry",
    "平静": "calm",
    "困倦": "sleepy",
    "淡定": "composed",
    "沉静": "serene",
    "平和": "peaceful",
    "诧异": "astonishment",
    "自信": "confident",
    "专横": "overbearing",
    "渺小": "insignificant",
}


def _avg_vad(entries: List[Tuple[float, float, float]]) -> Dict[str, float]:
    n = len(entries)
    v = sum(e[0] for e in entries) / n
    a = sum(e[1] for e in entries) / n
    d = sum(e[2] for e in entries) / n
    return {"valence": round(v, 3), "arousal": round(a, 3), "dominance": round(d, 3)}


def load_nrc_chinese(path: Path) -> Dict[str, List[Tuple[float, float, float]]]:
    zh_map: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            v, a, d = float(parts[1]), float(parts[2]), float(parts[3])
            zh_map[parts[4]].append((v, a, d))
    return zh_map


def load_nrc_english(path: Path) -> Dict[str, Tuple[float, float, float]]:
    en_map: Dict[str, Tuple[float, float, float]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            en_map[parts[0].lower()] = (float(parts[1]), float(parts[2]), float(parts[3]))
    return en_map


def lookup_vad(
    word: str,
    zh_map: Dict[str, List[Tuple[float, float, float]]],
    en_map: Dict[str, Tuple[float, float, float]],
) -> Tuple[Optional[Dict[str, float]], str]:
    if word in zh_map:
        return _avg_vad(zh_map[word]), "nrc_zh"

    en_key = CHINESE_TO_ENGLISH_FALLBACK.get(word)
    if en_key and en_key in en_map:
        v, a, d = en_map[en_key]
        return {"valence": round(v, 3), "arousal": round(a, 3), "dominance": round(d, 3)}, f"nrc_en:{en_key}"

    return None, "missing"


def update_master_lexicon(
    master_path: Path = MASTER_LEXICON_PATH,
    zh_nrc_path: Path = NRC_ZH_PATH,
    en_nrc_path: Path = NRC_EN_PATH,
    dry_run: bool = False,
) -> Dict:
    if not zh_nrc_path.exists():
        raise FileNotFoundError(f"未找到简体中文 NRC 词表: {zh_nrc_path}")
    if not en_nrc_path.exists():
        raise FileNotFoundError(f"未找到英文 NRC 词表: {en_nrc_path}")
    if not master_path.exists():
        raise FileNotFoundError(f"未找到主词库: {master_path}")

    zh_map = load_nrc_chinese(zh_nrc_path)
    en_map = load_nrc_english(en_nrc_path)

    with open(master_path, encoding="utf-8") as f:
        lexicon = json.load(f)

    stats = {"nrc_zh": 0, "nrc_en": 0, "missing": 0}
    report = []

    new_words = {}
    for word, old_vals in lexicon["words"].items():
        vad, source = lookup_vad(word, zh_map, en_map)
        if vad is None:
            stats["missing"] += 1
            new_words[word] = old_vals
            report.append((word, "missing", old_vals, old_vals))
            continue

        if source == "nrc_zh":
            stats["nrc_zh"] += 1
        else:
            stats["nrc_en"] += 1
        new_words[word] = vad
        report.append((word, source, old_vals, vad))

    lexicon["description"] = (
        "中文VAD情感词库 - 主词库（NRC VAD 官方值校准，0~1 区间，供各模型筛选单 token 极词）"
    )
    lexicon["format"]["valence"] = "效价，NRC 官方 0~1，AdaptiveScaler 映射至 [-1,1]"
    lexicon["format"]["arousal"] = "唤醒度，NRC 官方 0~1，AdaptiveScaler 映射至 [-1,1]"
    lexicon["format"]["dominance"] = "支配度，NRC 官方 0~1，AdaptiveScaler 映射至 [-1,1]"
    lexicon["words"] = new_words
    lexicon["nrc_calibration"] = {
        "source_zh": str(zh_nrc_path.relative_to(project_root)),
        "source_en": str(en_nrc_path.relative_to(project_root)),
        "stats": stats,
    }

    if not dry_run:
        with open(master_path, "w", encoding="utf-8") as f:
            json.dump(lexicon, f, indent=2, ensure_ascii=False)

    print(f"主词库: {master_path}")
    print(f"  NRC 简体中文直接匹配: {stats['nrc_zh']}")
    print(f"  NRC 英文回退: {stats['nrc_en']}")
    print(f"  未匹配（保留原值）: {stats['missing']}")
    if stats["missing"]:
        missing_words = [w for w, src, _, _ in report if src == "missing"]
        print(f"  未匹配词: {missing_words}")

    print("\n部分更新示例（旧 -> 新）:")
    for word, source, old, new in report[:8]:
        print(f"  {word} [{source}]: V {old['valence']:.3f}->{new['valence']:.3f}, "
              f"A {old['arousal']:.3f}->{new['arousal']:.3f}, D {old['dominance']:.3f}->{new['dominance']:.3f}")

    return lexicon


def main():
    parser = argparse.ArgumentParser(description="用 NRC 官方 VAD 值更新主词库")
    parser.add_argument("--dry-run", action="store_true", help="只预览，不写文件")
    args = parser.parse_args()
    update_master_lexicon(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
