"""
NIPS B1 极词 / hold-out 词表逻辑（中/英共用同一套规则）。

- NRC 分数统一为 bipolar [-1, 1]
- 极词：PRIORITY 情感种子 → NRC 主导维 + ±0.45 → Qwen 单 token
- Hold-out：300 词（100/维），主导维分配 + 情感过滤 + 单 token，与极词 disjoint
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

POLE_ORDER: List[str] = [
    "valence_positive",
    "valence_negative",
    "arousal_high",
    "arousal_low",
    "dominance_high",
    "dominance_low",
]

DIM_NAMES = ("valence", "arousal", "dominance")
N_HOLDOUT_PER_AXIS = 100
EMOTION_INTENSITY_THRESHOLD = 0.42
DOMINANT_AXIS_MIN = 0.22
POLE_EXTREMITY_THRESHOLD = 0.45

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

# NIPS Reform 英文情感种子（节选完整表见 finalV1 b1_reform_lexicon.py）
PRIORITY_EN: Dict[str, List[str]] = {
    "valence_positive": [
        "love", "joy", "happy", "bliss", "glad", "delight", "hope", "peace", "cheerful",
        "wonderful", "great", "pleased", "grateful", "proud", "lucky", "fantastic",
        "excellent", "sweet", "fun", "lovely", "joyful", "delightful", "merry",
        "elated", "ecstatic", "euphoric", "serene", "content", "satisfied", "blessed",
        "beautiful", "nice", "good", "perfect", "amazing", "awesome", "pleasant",
        "calm", "harmony", "kindness", "passion", "smile", "celebrate", "victory",
        "success", "winner", "hero", "honor", "comfort", "safe", "trust", "free",
    ],
    "valence_negative": [
        "sad", "hate", "grief", "anger", "fear", "pain", "death", "evil", "loss",
        "hurt", "harm", "lonely", "afraid", "worried", "angry", "mad", "upset",
        "depressed", "miserable", "unhappy", "horrible", "terrible", "awful", "nasty",
        "cruel", "violent", "murder", "war", "hell", "disaster", "tragedy", "failure",
        "regret", "shame", "guilt", "anxiety", "stress", "panic", "terror", "nightmare",
        "horror", "disgust", "jealous", "bitter", "torture", "suffer", "destroy", "ruin",
        "lost", "alone", "abandoned", "helpless", "hopeless", "worthless", "enemy", "rage",
    ],
    "arousal_high": [
        "rage", "fury", "panic", "terror", "alarm", "stress", "excited", "thrill", "wild",
        "intense", "fierce", "violent", "explosive", "nervous", "tense", "restless",
        "agitated", "frenzy", "urgent", "rush", "scream", "shout", "fight", "chaos",
        "frantic", "surprise", "anxious", "wrath", "irate", "enraged", "outraged",
        "manic", "thrilling", "jittery", "heated", "burning", "feverish",
    ],
    "arousal_low": [
        "calm", "quiet", "still", "sleep", "rest", "relax", "mellow", "soft", "slow",
        "lazy", "dull", "numb", "sluggish", "drowsy", "sleepy", "nap", "placid",
        "serene", "tranquil", "peaceful", "passive", "inactive", "stable", "steady",
        "silent", "subdued", "gentle", "ease", "weary", "fatigue", "tired", "exhausted",
        "listless", "unmoving", "unhurried", "stoic",
    ],
    "dominance_high": [
        "power", "control", "dominate", "master", "boss", "leader", "command", "authority",
        "strength", "mighty", "powerful", "bold", "brave", "confident", "proud", "pride",
        "victory", "win", "winner", "champion", "hero", "king", "ruler", "rule", "rich",
        "success", "expert", "genius", "smart", "ambitious", "assertive", "force",
        "formidable", "potent", "influential", "prestige", "royal", "dominant",
    ],
    "dominance_low": [
        "weak", "small", "meek", "humble", "inferior", "submissive", "submit", "slave",
        "victim", "helpless", "powerless", "fragile", "frail", "feeble", "yield",
        "surrender", "defeat", "broken", "pathetic", "poor", "plead", "cower", "shy",
        "timid", "fearful", "scared", "insignificant", "worthless", "useless", "hopeless",
        "humiliated", "shame", "loser", "failure", "reject", "nothing", "nobody", "empty",
        "passive", "dependent", "low", "weakness",
    ],
}

# 中文情感种子（主词库 + 常见情绪词；与 PRIORITY_EN 同结构）
PRIORITY_ZH: Dict[str, List[str]] = {
    "valence_positive": [
        "喜悦", "快乐", "幸福", "开心", "高兴", "愉快", "满足", "兴奋", "胜利", "成功",
        "热爱", "喜爱", "感激", "感恩", "和谐", "完美", "希望", "期待", "自豪", "骄傲",
        "舒适", "温暖", "甜蜜", "美好", "欣慰", "赞美", "友好", "狂喜", "和谐", "荣耀",
    ],
    "valence_negative": [
        "悲伤", "痛苦", "绝望", "失望", "沮丧", "难过", "哀伤", "忧郁", "憎恨", "厌恶",
        "讨厌", "失败", "孤独", "寂寞", "不幸", "折磨", "心碎", "失落", "凄惨", "反感",
        "嫌弃", "厌烦", "煎熬", "悲戚", "惆怅",
    ],
    "arousal_high": [
        "愤怒", "暴怒", "狂怒", "生气", "恼火", "激动", "亢奋", "振奋", "紧张", "爆发",
        "恐惧", "害怕", "惊恐", "惊慌", "焦虑", "心慌", "震惊", "惊讶", "兴奋", "愤慨",
    ],
    "arousal_low": [
        "平静", "放松", "宁静", "安详", "淡定", "沉静", "平和", "麻木", "困倦", "慵懒",
        "迟缓", "稳定", "舒适", "安逸", "被动", "服从", "和谐", "满足",
    ],
    "dominance_high": [
        "支配", "控制", "征服", "强大", "权威", "自信", "胜利", "骄傲", "荣耀", "自豪",
        "成功", "完美", "威严", "专横", "傲慢",
    ],
    "dominance_low": [
        "无助", "脆弱", "服从", "被动", "卑微", "畏缩", "顺从", "怯懦", "渺小", "屈服",
        "失败", "绝望", "失落", "孤独", "寂寞",
    ],
}

# 法语情感种子（NRC 法译 + 常用情绪词；与 PRIORITY_EN 同结构）
PRIORITY_FR: Dict[str, List[str]] = {
    "valence_positive": [
        "amour", "joie", "heureux", "bonheur", "content", "rire", "paix", "espoir",
        "merveilleux", "excellent", "doux", "génial", "chance", "fierté", "gratitude",
        "sérénité", "harmonie", "bon", "parfait", "magnifique", "chaleur", "amitié",
        "victoire", "succès", "fête", "célébration", "beauté", "plaisir", "bonjour",
        "fortune", "ange", "ami", "fruit", "concert", "sport", "animation", "cool",
    ],
    "valence_negative": [
        "triste", "haine", "deuil", "colère", "peur", "douleur", "mort", "mal", "perte",
        "blessure", "seul", "inquiet", "fâché", "déprimé", "misérable", "malheureux",
        "horrible", "terrible", "affreux", "cruel", "violent", "guerre", "tragédie",
        "échec", "regret", "honte", "culpabilité", "anxiété", "stress", "panique",
        "terreur", "cauchemar", "horreur", "dégoût", "jalousie", "amer", "souffrir",
        "détruire", "perdu", "abandonné", "désespoir", "ennemi", "rage", "pire", "vil",
        "grave", "orage", "erreur", "sale", "privation",
    ],
    "arousal_high": [
        "rage", "fureur", "panique", "terreur", "alarme", "stress", "excité", "frisson",
        "intense", "féroce", "violent", "explosif", "nerveux", "tendu", "agité",
        "frénésie", "urgent", "urgence", "cri", "combat", "chaos", "frénétique",
        "surprise", "anxieux", "colère", "furieux", "outré", "manique", "palpitant",
        "danger", "action", "impact", "crime", "extrême", "ovation", "sexuel",
    ],
    "arousal_low": [
        "calme", "quiet", "repos", "sommeil", "détente", "doux", "lent", "paresseux",
        "engourdi", "somnolent", "sieste", "placide", "serein", "tranquille", "paisible",
        "passif", "inactif", "stable", "silence", "doux", "fatigue", "épuisé",
        "indolent", "zen", "pause", "patient", "minimal", "routine", "thé", "lit",
    ],
    "dominance_high": [
        "pouvoir", "contrôle", "dominer", "maître", "chef", "leader", "commandement",
        "autorité", "force", "fort", "puissant", "audacieux", "courageux", "confiant",
        "fierté", "victoire", "champion", "héros", "roi", "riche", "succès",
        "expert", "génie", "intelligent", "ambitieux", "assertif", "formidable",
        "influent", "prestige", "royal", "dominant", "prof", "parent", "nation",
    ],
    "dominance_low": [
        "faible", "petit", "humble", "inférieur", "soumis", "esclave", "victime",
        "impuissant", "fragile", "frêle", "faiblesse", "capituler", "défaite",
        "pathétique", "pauvre", "timide", "peureux", "insignifiant", "inutile",
        "désespoir", "humilié", "honte", "perdant", "échec", "rejet", "vide", "fade",
        "bas", "ombre", "sans", "non", "micro", "mite", "mono",
    ],
}

# 日语情感种子（NRC 日译 + 常用情绪词；与 PRIORITY_EN 同结构）
PRIORITY_JA: Dict[str, List[str]] = {
    "valence_positive": [
        "愛", "喜び", "幸福", "楽しい", "嬉しい", "満足", "興奮", "勝利", "成功",
        "感謝", "調和", "完璧", "希望", "期待", "誇り", "自信", "快適", "温かい",
        "甘い", "美しい", "祝福", "友好", "狂喜", "栄光", "すごい", "良い", "健康",
        "利益", "楽しみ", "名人", "恋人", "家族", "思い出", "美味しい", "心", "空",
    ],
    "valence_negative": [
        "悲しみ", "苦痛", "絶望", "失望", "落胆", "悲しい", "憂鬱", "憎しみ", "嫌悪",
        "嫌い", "失敗", "孤独", "不幸", "苦しみ", "喪失", "悲惨", "反感", "嫌う", "退屈",
        "死", "混沌", "暴力", "毒", "危害", "負け", "嘘", "困難", "感染", "罰", "弱点",
        "恐怖", "痛み", "不良", "毒性", "事故", "出血", "癌", "窒息", "悪い", "棺",
    ],
    "arousal_high": [
        "怒り", "激怒", "狂怒", "怒る", "苛立ち", "興奮", "亢進", "緊張", "爆発",
        "恐れ", "怖い", "恐怖", "驚愕", "驚き", "不安", "慌て", "衝撃", "憤慨",
        "性的", "逮捕", "火力", "怪物", "侵略", "野生", "急", "事件", "追求",
        "革命", "地震", "防御", "討論", "大胆", "犯罪", "兵器", "暴露", "火山",
    ],
    "arousal_low": [
        "平静", "リラックス", "静けさ", "安らぎ", "落ち着き", "穏やか", "鈍い", "眠い",
        "だるい", "遅い", "安定", "快適", "安楽", "受動", "従順", "調和", "満足",
        "睡眠", "正常", "水", "一般", "標準", "微妙", "椅子", "カメ", "葡萄", "紙",
    ],
    "dominance_high": [
        "支配", "コントロール", "征服", "強い", "権威", "自信", "勝利", "誇り", "栄光",
        "成功", "完璧", "威厳", "傲慢", "成功", "パワー", "帝国", "指令", "大使",
        "全能", "王国", "家長", "知識", "キング", "大臣", "独立", "女神", "巨人",
        "首相", "先生", "力", "管理者",
    ],
    "dominance_low": [
        "無力", "脆弱", "従順", "受動", "卑しい", "恐縮", "順従", "臆病", "微力",
        "屈服", "失敗", "絶望", "喪失", "孤独", "以下", "最低", "真空", "小さな",
        "最小", "不合格", "虚弱", "小さい", "低い", "不要", "少数", "不用", "泥",
        "羊", "灰", "滴", "粉", "蛾", "蚊",
    ],
}


def to_bipolar(v: float, a: float, d: float) -> Dict[str, float]:
    return {
        "valence": 2.0 * v - 1.0,
        "arousal": 2.0 * a - 1.0,
        "dominance": 2.0 * d - 1.0,
    }


def load_nrc_english_bipolar(path: Path) -> Dict[str, Dict[str, float]]:
    """BipolarScale/NRC-VAD-Lexicon.txt，仅无空格 unigram。"""
    out: Dict[str, Dict[str, float]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            w = parts[0].strip().lower()
            if not w or " " in w:
                continue
            try:
                out[w] = {
                    "valence": float(parts[1]),
                    "arousal": float(parts[2]),
                    "dominance": float(parts[3]),
                }
            except ValueError:
                continue
    return out


def load_nrc_chinese_bipolar(path: Path) -> Dict[str, Dict[str, float]]:
    """Chinese-Simplified-NRC-VAD-Lexicon.txt；0~1 → [-1,1]，同形中文取均值。"""
    acc: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            zh = parts[4].strip()
            if not zh or " " in zh or not _CJK_RE.search(zh):
                continue
            try:
                acc[zh].append((float(parts[1]), float(parts[2]), float(parts[3])))
            except ValueError:
                continue
    out: Dict[str, Dict[str, float]] = {}
    for zh, vals in acc.items():
        n = len(vals)
        v = sum(x[0] for x in vals) / n
        a = sum(x[1] for x in vals) / n
        d = sum(x[2] for x in vals) / n
        out[zh] = to_bipolar(v, a, d)
    return out


def extend_priority_from_nrc(
    priority: Dict[str, List[str]],
    nrc: Dict[str, Dict[str, float]],
    top_k: int = 120,
) -> Dict[str, List[str]]:
    """用 NRC 极值排序扩充各极候选（与 NIPS fallback 一致）。"""
    merged = {p: list(priority.get(p, [])) for p in POLE_ORDER}
    for pole in POLE_ORDER:
        extras = _nrc_fallback_order(pole, nrc)[:top_k]
        merged[pole] = _dedupe_preserve(merged[pole] + extras)
    return merged


def dominant_dim_idx(vad: Dict[str, float]) -> int:
    av = [abs(vad["valence"]), abs(vad["arousal"]), abs(vad["dominance"])]
    best = 0
    for i in range(1, 3):
        if av[i] > av[best]:
            best = i
    return best


def pole_predicate(pole: str, vad: Dict[str, float]) -> bool:
    d = dominant_dim_idx(vad)
    v, a, dom = vad["valence"], vad["arousal"], vad["dominance"]
    if pole == "valence_positive":
        return d == 0 and v > POLE_EXTREMITY_THRESHOLD
    if pole == "valence_negative":
        return d == 0 and v < -POLE_EXTREMITY_THRESHOLD
    if pole == "arousal_high":
        return d == 1 and a > POLE_EXTREMITY_THRESHOLD
    if pole == "arousal_low":
        return d == 1 and a < -POLE_EXTREMITY_THRESHOLD
    if pole == "dominance_high":
        return d == 2 and dom > POLE_EXTREMITY_THRESHOLD
    if pole == "dominance_low":
        return d == 2 and dom < -POLE_EXTREMITY_THRESHOLD
    raise ValueError(pole)


def extremity_key(pole: str, vad: Dict[str, float]) -> float:
    v, a, dom = vad["valence"], vad["arousal"], vad["dominance"]
    if pole.startswith("valence"):
        return abs(v)
    if pole.startswith("arousal"):
        return abs(a)
    return abs(dom)


def _dedupe_preserve(seq: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for w in seq:
        w = w.strip()
        if not w or w in seen:
            continue
        seen.add(w)
        out.append(w)
    return out


def _nrc_fallback_order(pole: str, nrc: Dict[str, Dict[str, float]]) -> List[str]:
    scored: List[Tuple[str, float]] = []
    for w, vad in nrc.items():
        if pole_predicate(pole, vad):
            scored.append((w, extremity_key(pole, vad)))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return [w for w, _ in scored]


def emotion_seed_words(language: str) -> Set[str]:
    pri = PRIORITY_EN if language == "en" else PRIORITY_ZH
    s: Set[str] = set()
    for words in pri.values():
        for w in words:
            s.add(w.strip().lower() if language == "en" else w.strip())
    return s


def polarity_intensity(vad: Dict[str, float]) -> float:
    return max(abs(vad["valence"]), abs(vad["arousal"]), abs(vad["dominance"]))


def emotion_ok(word: str, vad: Dict[str, float], seed: Set[str]) -> bool:
    if word in seed:
        return True
    return polarity_intensity(vad) >= EMOTION_INTENSITY_THRESHOLD


def is_single_token(word: str, tokenizer) -> bool:
    return len(tokenizer.encode(word, add_special_tokens=False)) == 1


def select_word_groups(
    pole_n: int,
    tokenizer,
    nrc: Dict[str, Dict[str, float]],
    priority: Dict[str, List[str]],
    exclude: Optional[Set[str]] = None,
    allow_cross_pole_reuse: bool = False,
) -> Dict[str, List[str]]:
    exclude = exclude or set()
    used: Set[str] = set()
    groups: Dict[str, List[str]] = {p: [] for p in POLE_ORDER}

    def try_add(pole: str, w: str) -> bool:
        key = w.strip().lower() if w.isascii() else w.strip()
        if key in exclude or key not in nrc:
            return False
        if not allow_cross_pole_reuse and key in used:
            return False
        if key in groups[pole]:
            return False
        if not pole_predicate(pole, nrc[key]):
            return False
        if not is_single_token(key, tokenizer):
            return False
        groups[pole].append(key)
        if not allow_cross_pole_reuse:
            used.add(key)
        return True

    for pole in POLE_ORDER:
        for w in _dedupe_preserve(priority.get(pole, [])):
            if len(groups[pole]) >= pole_n:
                break
            try_add(pole, w)

    for pole in POLE_ORDER:
        if len(groups[pole]) >= pole_n:
            continue
        for w in _nrc_fallback_order(pole, nrc):
            if len(groups[pole]) >= pole_n:
                break
            if w in exclude:
                continue
            if not allow_cross_pole_reuse and w in used:
                continue
            try_add(pole, w)

    short = [p for p in POLE_ORDER if len(groups[p]) < pole_n]
    if short:
        raise RuntimeError(
            f"Could not fill poles {short} to n={pole_n}. "
            f"Got: {[(p, len(groups[p])) for p in POLE_ORDER]}"
        )
    for p in POLE_ORDER:
        groups[p] = groups[p][:pole_n]
    return groups


def build_lexicon_payload(
    word_groups: Dict[str, List[str]],
    nrc: Dict[str, Dict[str, float]],
    language: str,
) -> dict:
    words: Dict[str, Dict[str, float]] = {}
    for pole, ws in word_groups.items():
        for w in ws:
            words[w] = dict(nrc[w])
    return {
        "description": f"E1 pole lexicon ({language}); NIPS B1 rules; bipolar [-1,1]",
        "language": language,
        "normalized_to_minus_one_one": True,
        "word_groups": word_groups,
        "words": words,
    }


def build_corpus_lines(word_groups: Dict[str, List[str]], language: str) -> List[str]:
    lines: List[str] = []
    for pole in POLE_ORDER:
        for w in word_groups[pole]:
            if language == "zh":
                lines.append(f"我今天感到{w}。")
                lines.append(
                    f"经历了上周发生的一切，我仍然对这种情况感到{w}，"
                    f"无法停止去想这件事。"
                )
            else:
                # en/fr/ja/…：词放句首，便于 Qwen 单 token 与 Scaler 匹配
                lines.append(f"{w} is how I feel today.")
                lines.append(
                    f"{w} is exactly what I feel after everything that happened last week "
                    f"and I cannot stop thinking about it."
                )
    return lines


def build_holdout_lexicon(
    tokenizer,
    nrc: Dict[str, Dict[str, float]],
    language: str,
    n_per_axis: int = N_HOLDOUT_PER_AXIS,
) -> dict:
    """300 词 hold-out：100/维，主导维 + 情感 + 单 token，三轴互不重复。"""
    seed = emotion_seed_words(language)
    buckets: Dict[str, List[Tuple[str, Dict[str, float]]]] = {
        d: [] for d in DIM_NAMES
    }
    for w, vad in nrc.items():
        if not emotion_ok(w, vad, seed):
            continue
        d_idx = dominant_dim_idx(vad)
        dim = DIM_NAMES[d_idx]
        if abs(vad[dim]) < DOMINANT_AXIS_MIN:
            continue
        buckets[dim].append((w, vad))

    for dim in DIM_NAMES:
        buckets[dim].sort(key=lambda t: (-abs(t[1][dim]), t[0]))

    per_axis: Dict[str, Dict[str, Dict[str, float]]] = {}
    used: Set[str] = set()
    for dim in DIM_NAMES:
        picked: Dict[str, Dict[str, float]] = {}
        for w, vad in buckets[dim]:
            if w in used:
                continue
            if not is_single_token(w, tokenizer):
                continue
            picked[w] = dict(vad)
            used.add(w)
            if len(picked) >= n_per_axis:
                break
        if len(picked) < n_per_axis:
            raise RuntimeError(
                f"{language}: only {len(picked)} hold-out words for {dim} "
                f"(need {n_per_axis})"
            )
        per_axis[dim] = picked

    words_all = {}
    for dim in DIM_NAMES:
        words_all.update(per_axis[dim])

    return {
        "description": (
            f"E1 hold-out Spearman set ({language}, Qwen single-token). "
            f"{n_per_axis} words per axis, dominant-dim assignment, emotion filter."
        ),
        "language": language,
        "normalized_to_minus_one_one": True,
        "n_per_axis": n_per_axis,
        "per_axis": per_axis,
        "words": words_all,
        "axis_names": list(DIM_NAMES),
    }


def get_nrc_paths(project_root: Path) -> Tuple[Path, Path]:
    nrc_dir = project_root / "data" / "NRC-VAD-Lexicon"
    en = nrc_dir / "BipolarScale" / "NRC-VAD-Lexicon.txt"
    zh = nrc_dir / "OneFilePerLanguage" / "Chinese-Simplified-NRC-VAD-Lexicon.txt"
    return en, zh


def load_language_nrc(language: str, project_root: Path) -> Dict[str, Dict[str, float]]:
    en_path, zh_path = get_nrc_paths(project_root)
    if language == "en":
        return load_nrc_english_bipolar(en_path)
    if language == "zh":
        return load_nrc_chinese_bipolar(zh_path)
    raise ValueError(language)


def get_priority(language: str, nrc: Dict[str, Dict[str, float]]) -> Dict[str, List[str]]:
    base = PRIORITY_EN if language == "en" else PRIORITY_ZH
    return extend_priority_from_nrc(base, nrc)
