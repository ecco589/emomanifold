#!/usr/bin/env python3
"""
VAD词库准备脚本
加载并验证VAD词库
"""
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import config

def load_vad_lexicon(lexicon_path=None):
    """
    加载VAD词库
    
    Args:
        lexicon_path: 词库文件路径，默认使用config中的路径
        
    Returns:
        Dict: 词库数据
    """
    if lexicon_path is None:
        lexicon_path = config.VAD_LEXICON_DIR / "vad_lexicon_chinese.json"
    
    lexicon_path = Path(lexicon_path)
    
    if not lexicon_path.exists():
        print(f"词库文件不存在: {lexicon_path}")
        return None
    
    try:
        with open(lexicon_path, 'r', encoding='utf-8') as f:
            lexicon = json.load(f)
        print(f"成功加载词库: {lexicon_path}")
        print(f"   包含 {len(lexicon.get('words', {}))} 个情感词")
        return lexicon
    except Exception as e:
        print(f"加载词库失败: {e}")
        return None

def get_word_groups(lexicon):
    """
    从词库中提取词组
    
    Args:
        lexicon: 词库数据
        
    Returns:
        Dict: 包含正负效价词、高低唤醒度词的字典
    """
    if lexicon is None:
        return None
    
    word_groups = lexicon.get('word_groups', {})
    
    return {
        'valence_positive': word_groups.get('valence_positive', []),
        'valence_negative': word_groups.get('valence_negative', []),
        'arousal_high': word_groups.get('arousal_high', []),
        'arousal_low': word_groups.get('arousal_low', [])
    }

if __name__ == "__main__":
    print("=" * 50)
    print("VAD词库准备脚本")
    print("=" * 50)
    
    lexicon = load_vad_lexicon()
    
    if lexicon:
        word_groups = get_word_groups(lexicon)
        print("\n词组统计:")
        print(f"  高效价词: {len(word_groups['valence_positive'])} 个")
        print(f"  低效价词: {len(word_groups['valence_negative'])} 个")
        print(f"  高唤醒度词: {len(word_groups['arousal_high'])} 个")
        print(f"  低唤醒度词: {len(word_groups['arousal_low'])} 个")
        
        print("\n示例词:")
        print(f"  高效价: {', '.join(word_groups['valence_positive'][:5])}")
        print(f"  低效价: {', '.join(word_groups['valence_negative'][:5])}")
        print(f"  高唤醒: {', '.join(word_groups['arousal_high'][:5])}")
        print(f"  低唤醒: {', '.join(word_groups['arousal_low'][:5])}")
