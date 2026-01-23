"""
生成训练语料库脚本
生成包含词典词的句子，涵盖多种情绪场景，包含短、中、长篇幅语料
"""
import json
from pathlib import Path
import random


def load_lexicon():
    """加载词典"""
    lexicon_path = Path(__file__).parent.parent / "data" / "vad_lexicon" / "vad_lexicon_chinese.json"
    with open(lexicon_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_corpus():
    """生成训练语料库"""
    lexicon = load_lexicon()
    words = lexicon['words']
    
    # 按情绪类型组织词典词
    positive_words = lexicon['word_groups']['valence_positive']
    negative_words = lexicon['word_groups']['valence_negative']
    high_arousal_words = lexicon['word_groups']['arousal_high']
    low_arousal_words = lexicon['word_groups']['arousal_low']
    
    # 短句模板（10-20字）
    short_templates = {
        'positive_high': [
            "我赢得了比赛，现在感到非常{word}",
            "今天收到了好消息，让我非常{word}",
            "看到这个结果，我真的很{word}",
            "听到这个消息，我感到无比{word}",
            "这个成就让我感到非常{word}",
        ],
        'positive_low': [
            "今天过得很{word}",
            "我感到很{word}",
            "现在的心情很{word}",
            "我对现状感到{word}",
            "这让我感到{word}",
        ],
        'negative_high': [
            "这件事让我感到非常{word}",
            "我现在很{word}",
            "这让我感到极其{word}",
            "我对这件事感到{word}",
            "这让我感到非常{word}",
        ],
        'negative_low': [
            "我现在感到{word}",
            "这让我感到{word}",
            "我对现状感到{word}",
            "我现在很{word}",
            "这让我觉得很{word}",
        ],
        'surprise': [
            "看到这个结果，我感到非常{word}",
            "这让我感到{word}",
            "我对这件事感到{word}",
            "这让我感到非常{word}",
            "我现在很{word}",
        ],
        'fear': [
            "我现在感到{word}",
            "这让我感到{word}",
            "我对这件事感到{word}",
            "这让我感到非常{word}",
            "我现在很{word}",
        ],
        'anger': [
            "这件事让我感到{word}",
            "我现在很{word}",
            "这让我感到非常{word}",
            "我对这件事感到{word}",
            "这让我感到极其{word}",
        ],
        'calm': [
            "我现在感到{word}",
            "这让我感到{word}",
            "我对现状感到{word}",
            "我现在很{word}",
            "这让我觉得很{word}",
        ],
    }
    
    # 中句模板（20-40字）
    medium_templates = {
        'positive_high': [
            "当我听到这个消息的时候，内心感到非常{word}，因为这是我期待已久的结果",
            "经过长时间的努力，我终于完成了这个目标，现在感到特别{word}和满足",
            "看到朋友们为我庆祝，我的心情变得非常{word}，这种喜悦难以言表",
            "在经历了这么多困难之后，终于迎来了好消息，让我感到无比{word}",
            "这个意外的惊喜让我感到非常{word}，我完全没有想到会有这样的结果",
        ],
        'positive_low': [
            "今天的天气很好，阳光洒在身上，让我感到非常{word}和舒适",
            "周末的午后，我坐在窗边看书，心情很{word}，享受着这份宁静",
            "完成了所有的工作之后，我现在感到{word}，可以好好休息一下了",
            "看到家人健康快乐，我的内心感到{word}，这就是最大的幸福",
            "在这个安静的夜晚，我感到很{word}，所有的烦恼都暂时放下了",
        ],
        'negative_high': [
            "当我得知这个消息的时候，内心感到非常{word}，完全无法接受这个事实",
            "面对这个突如其来的变故，我现在感到极其{word}，不知道该如何应对",
            "看到这个令人失望的结果，我的心情变得非常{word}，充满了愤怒和不满",
            "经历了这么多挫折之后，我现在感到{word}，感觉所有的努力都白费了",
            "这个意外的打击让我感到非常{word}，完全没有预料到会有这样的结果",
        ],
        'negative_low': [
            "最近发生的事情让我感到{word}，心情一直很低落，不知道什么时候才能好转",
            "面对这个无法改变的现实，我现在感到{word}，只能默默地接受这一切",
            "失去了重要的东西之后，我的内心感到{word}，感觉生活失去了意义",
            "在这个孤独的夜晚，我感到{word}，所有的希望似乎都离我远去了",
            "经历了这么多困难，我现在感到{word}，不知道未来会是什么样子",
        ],
        'surprise': [
            "当我看到这个结果的时候，感到非常{word}，完全没有想到会是这样的结局",
            "这个消息让我感到{word}，我站在原地愣了很久，不敢相信这是真的",
            "面对这个意外的发现，我现在感到{word}，完全超出了我的预期",
            "看到这个令人震惊的场景，我的心情变得非常{word}，一时之间无法反应",
            "这个突如其来的消息让我感到{word}，我完全没有心理准备",
        ],
        'fear': [
            "当我面对这个未知的情况时，内心感到{word}，不知道接下来会发生什么",
            "看到这个危险的信号，我现在感到非常{word}，担心会有不好的事情发生",
            "在这个黑暗的环境中，我感到{word}，总觉得有什么不好的事情要发生",
            "面对这个令人不安的情况，我的心情变得{word}，充满了担忧和焦虑",
            "这个不确定的未来让我感到{word}，我不知道自己能否应对这些挑战",
        ],
        'anger': [
            "当我看到这个不公平的对待时，内心感到非常{word}，完全无法接受",
            "面对这个令人愤怒的行为，我现在感到{word}，想要立即采取行动",
            "经历了这么多委屈之后，我现在感到{word}，所有的怒火都涌上心头",
            "看到这个令人恼火的结果，我的心情变得非常{word}，完全无法平静",
            "这个不公正的决定让我感到{word}，我觉得自己受到了极大的伤害",
        ],
        'calm': [
            "在这个安静的午后，我感到{word}，所有的烦恼都暂时消失了",
            "听着轻柔的音乐，我现在感到{word}，心情变得非常平静和舒适",
            "完成了所有的工作之后，我感到{word}，可以好好享受这份宁静了",
            "在这个美丽的环境中，我的心情变得{word}，所有的压力都释放了",
            "看到这个宁静的景色，我感到{word}，内心充满了平和与满足",
        ],
    }
    
    # 长句模板（40-80字）
    long_templates = {
        'positive_high': [
            "当我终于完成了这个艰巨的任务，看到成果的那一刻，内心感到非常{word}，所有的努力和付出都得到了回报，这种成就感让我觉得一切都是值得的",
            "经过长时间的等待和努力，我终于等到了这个好消息，现在感到特别{word}，这种喜悦之情难以言表，我想要和所有人分享这份快乐",
            "看到朋友们为我庆祝，听到他们的祝福，我的心情变得非常{word}，这种被认可和关爱的感觉让我感到无比幸福和满足",
            "在经历了这么多困难和挫折之后，我终于迎来了这个重要的时刻，现在感到非常{word}，所有的坚持都没有白费",
            "这个意外的惊喜让我感到{word}，我完全没有想到会有这样的结果，这种突如其来的快乐让我觉得生活充满了希望",
        ],
        'positive_low': [
            "今天的天气特别好，阳光温暖地洒在身上，我坐在窗边看着外面的风景，心情很{word}，享受着这份难得的宁静和舒适",
            "完成了所有的工作之后，我现在感到{word}，可以好好休息一下了，这种轻松的感觉让我觉得生活很美好",
            "看到家人健康快乐地生活，我的内心感到{word}，这就是我最大的幸福，没有什么比家人的健康更重要了",
            "在这个安静的夜晚，我独自一人坐在阳台上，看着星空，感到{word}，所有的烦恼都暂时放下了，内心充满了平静",
            "周末的午后，我泡了一杯茶，坐在沙发上看着喜欢的书，心情很{word}，这种简单的生活让我感到满足和快乐",
        ],
        'negative_high': [
            "当我得知这个令人震惊的消息时，内心感到非常{word}，完全无法接受这个事实，我站在原地愣了很久，不知道该如何应对这个突如其来的变故",
            "面对这个令人失望的结果，我现在感到极其{word}，所有的努力似乎都白费了，这种挫败感让我感到非常沮丧和愤怒",
            "看到这个不公平的对待，我的心情变得非常{word}，我觉得自己受到了极大的伤害，这种愤怒和不满让我无法平静下来",
            "经历了这么多挫折和困难之后，我现在感到{word}，感觉所有的希望都破灭了，不知道未来会是什么样子",
            "这个意外的打击让我感到非常{word}，完全没有预料到会有这样的结果，这种突如其来的痛苦让我感到绝望",
        ],
        'negative_low': [
            "最近发生的事情让我感到{word}，心情一直很低落，不知道什么时候才能好转，这种忧郁的情绪一直困扰着我",
            "面对这个无法改变的现实，我现在感到{word}，只能默默地接受这一切，这种无力感让我感到非常沮丧",
            "失去了重要的东西之后，我的内心感到{word}，感觉生活失去了意义，不知道该如何继续下去",
            "在这个孤独的夜晚，我感到{word}，所有的希望似乎都离我远去了，这种绝望的感觉让我感到非常痛苦",
            "经历了这么多困难，我现在感到{word}，不知道未来会是什么样子，这种不确定感让我感到焦虑和担忧",
        ],
        'surprise': [
            "当我看到这个完全出乎意料的结果时，感到非常{word}，我站在原地愣了很久，完全不敢相信这是真的，这种震惊让我一时之间无法反应",
            "这个消息让我感到{word}，我完全没有心理准备，这种突如其来的发现完全超出了我的预期，让我感到非常意外",
            "面对这个令人震惊的场景，我的心情变得非常{word}，我从来没有想过会有这样的结果，这种惊讶让我感到不知所措",
            "看到这个意外的发现，我现在感到{word}，完全超出了我的想象，这种突如其来的惊喜让我感到非常兴奋",
            "这个令人震惊的消息让我感到{word}，我完全没有预料到会有这样的结果，这种意外让我感到既兴奋又紧张",
        ],
        'fear': [
            "当我面对这个未知的危险情况时，内心感到{word}，不知道接下来会发生什么，这种不确定感让我感到非常担忧和焦虑",
            "看到这个危险的信号，我现在感到非常{word}，担心会有不好的事情发生，这种恐惧让我感到非常不安",
            "在这个黑暗的环境中，我感到{word}，总觉得有什么不好的事情要发生，这种不安的感觉让我感到非常紧张",
            "面对这个令人不安的情况，我的心情变得{word}，充满了担忧和焦虑，不知道该如何应对这个未知的挑战",
            "这个不确定的未来让我感到{word}，我不知道自己能否应对这些挑战，这种恐惧和担忧让我感到非常焦虑",
        ],
        'anger': [
            "当我看到这个不公平的对待时，内心感到非常{word}，完全无法接受这个事实，这种愤怒让我想要立即采取行动来改变这个不公正的情况",
            "面对这个令人愤怒的行为，我现在感到{word}，所有的怒火都涌上心头，这种愤怒让我感到非常不满和愤慨",
            "经历了这么多委屈之后，我现在感到{word}，我觉得自己受到了极大的伤害，这种愤怒和不满让我无法平静下来",
            "看到这个令人恼火的结果，我的心情变得非常{word}，完全无法接受这个不公正的决定，这种愤怒让我感到非常痛苦",
            "这个不公正的决定让我感到{word}，我觉得自己受到了极大的伤害，这种愤怒和不满让我感到非常沮丧和失望",
        ],
        'calm': [
            "在这个安静的午后，我独自一人坐在窗边，看着外面的风景，感到{word}，所有的烦恼都暂时消失了，这种宁静让我感到非常舒适和满足",
            "听着轻柔的音乐，我现在感到{word}，心情变得非常平静和舒适，这种放松的感觉让我觉得生活很美好",
            "完成了所有的工作之后，我感到{word}，可以好好享受这份宁静了，这种轻松和满足让我感到非常快乐",
            "在这个美丽的环境中，我的心情变得{word}，所有的压力都释放了，这种平静和舒适让我感到非常满足",
            "看到这个宁静的景色，我感到{word}，内心充满了平和与满足，这种美好的感觉让我觉得生活充满了希望",
        ],
    }
    
    corpus = []
    
    # 情绪类型和对应的词典词
    # 注意：有些词可能同时属于多个类别，需要确保每个词都被分配到至少一个类别
    positive_high_words = [w for w in positive_words if w in high_arousal_words]
    positive_low_words = [w for w in positive_words if w in low_arousal_words]
    # 处理既不在high也不在low的positive词（应该很少）
    positive_other = [w for w in positive_words if w not in high_arousal_words and w not in low_arousal_words]
    if positive_other:
        # 根据arousal值分配到high或low
        for word in positive_other:
            if words[word]['arousal'] >= 0.5:
                positive_high_words.append(word)
            else:
                positive_low_words.append(word)
    
    negative_high_words = [w for w in negative_words if w in high_arousal_words]
    negative_low_words = [w for w in negative_words if w in low_arousal_words]
    # 处理既不在high也不在low的negative词
    negative_other = [w for w in negative_words if w not in high_arousal_words and w not in low_arousal_words]
    if negative_other:
        for word in negative_other:
            if words[word]['arousal'] >= 0.5:
                negative_high_words.append(word)
            else:
                negative_low_words.append(word)
    
    emotion_categories = [
        ('positive_high', positive_high_words),
        ('positive_low', positive_low_words),
        ('negative_high', negative_high_words),
        ('negative_low', negative_low_words),
        ('surprise', ['惊讶', '震惊', '诧异']),
        ('fear', ['恐惧', '害怕', '惊恐', '焦虑', '担忧']),
        ('anger', ['愤怒', '生气', '恼火', '暴怒', '愤慨']),
        ('calm', ['平静', '放松', '宁静', '安详', '困倦']),
    ]
    
    # 验证：确保所有词典词都被分配到至少一个类别
    all_categorized_words = set()
    for _, word_list in emotion_categories:
        all_categorized_words.update(word_list)
    
    uncategorized_words = set(words.keys()) - all_categorized_words
    if uncategorized_words:
        print(f"  警告：以下词未被分类: {uncategorized_words}")
        # 将它们添加到合适的类别
        for word in uncategorized_words:
            if word in positive_words:
                if words[word]['arousal'] >= 0.5:
                    emotion_categories[0][1].append(word)
                else:
                    emotion_categories[1][1].append(word)
            elif word in negative_words:
                if words[word]['arousal'] >= 0.5:
                    emotion_categories[2][1].append(word)
                else:
                    emotion_categories[3][1].append(word)
    
    # 目标：500句，平均每种情绪类型约62-63句
    # 分配策略：短句40%，中句35%，长句25%
    target_short = 200  # 40%
    target_medium = 175  # 35%
    target_long = 125   # 25%
    
    # 每种情绪类型的句子数
    sentences_per_category = 500 // len(emotion_categories)  # 约62-63句
    
    # 生成短句
    short_count = 0
    for category, word_list in emotion_categories:
        if not word_list:
            continue
        
        short_cat_templates = short_templates.get(category, [])
        if not short_cat_templates:
            continue
        
        # 每个类别生成一定数量的短句
        per_category_short = target_short // len(emotion_categories)
        for _ in range(per_category_short):
            if short_count >= target_short:
                break
            word = random.choice(word_list)
            template = random.choice(short_cat_templates)
            sentence = template.format(word=word)
            corpus.append(sentence)
            short_count += 1
    
    # 生成中句
    medium_count = 0
    for category, word_list in emotion_categories:
        if not word_list:
            continue
        
        medium_cat_templates = medium_templates.get(category, [])
        if not medium_cat_templates:
            continue
        
        # 每个类别生成一定数量的中句
        per_category_medium = target_medium // len(emotion_categories)
        for _ in range(per_category_medium):
            if medium_count >= target_medium:
                break
            word = random.choice(word_list)
            template = random.choice(medium_cat_templates)
            sentence = template.format(word=word)
            corpus.append(sentence)
            medium_count += 1
    
    # 生成长句
    long_count = 0
    for category, word_list in emotion_categories:
        if not word_list:
            continue
        
        long_cat_templates = long_templates.get(category, [])
        if not long_cat_templates:
            continue
        
        # 每个类别生成一定数量的长句
        per_category_long = target_long // len(emotion_categories)
        for _ in range(per_category_long):
            if long_count >= target_long:
                break
            word = random.choice(word_list)
            template = random.choice(long_cat_templates)
            sentence = template.format(word=word)
            corpus.append(sentence)
            long_count += 1
    
    # 确保每个词典词都有覆盖（补充）
    all_words_covered = set()
    for sentence in corpus:
        for word in words.keys():
            if word in sentence:
                all_words_covered.add(word)
    
    # 补充未覆盖的词（在补充到500之前完成）
    uncovered_words = set(words.keys()) - all_words_covered
    for word in uncovered_words:
        # 找到这个词所属的情绪类别
        found_category = None
        for category, word_list in emotion_categories:
            if word in word_list:
                found_category = category
                break
        
        if found_category:
            # 随机选择长度类型和模板
            length_type = random.choice(['short', 'medium', 'long'])
            if length_type == 'short':
                templates_dict = short_templates
            elif length_type == 'medium':
                templates_dict = medium_templates
            else:
                templates_dict = long_templates
            
            templates_list = templates_dict.get(found_category, [])
            if templates_list:
                template = random.choice(templates_list)
                sentence = template.format(word=word)
                corpus.append(sentence)
            else:
                # 如果没有模板，使用通用模板
                sentence = f"我现在感到{word}"
                corpus.append(sentence)
    
    # 打乱顺序
    random.shuffle(corpus)
    
    # 如果数量不足500，补充到500
    seen_sentences = set(corpus)
    max_attempts = 2000
    attempts = 0
    
    while len(corpus) < 500 and attempts < max_attempts:
        attempts += 1
        category, word_list = random.choice(emotion_categories)
        if not word_list:
            continue
        
        length_type = random.choice(['short', 'medium', 'long'])
        if length_type == 'short':
            templates_dict = short_templates
        elif length_type == 'medium':
            templates_dict = medium_templates
        else:
            templates_dict = long_templates
        
        templates_list = templates_dict.get(category, [])
        if templates_list:
            word = random.choice(word_list)
            template = random.choice(templates_list)
            sentence = template.format(word=word)
            if sentence not in seen_sentences:
                corpus.append(sentence)
                seen_sentences.add(sentence)
    
    # 如果超过500，先确保每个词至少有一句，然后随机选择
    if len(corpus) > 500:
        # 先确保每个词至少有一句
        final_corpus = []
        words_covered = set()
        
        # 第一遍：为每个词选择一句
        for word in words.keys():
            for sentence in corpus:
                if word in sentence and word not in words_covered:
                    final_corpus.append(sentence)
                    words_covered.add(word)
                    break
        
        # 第二遍：随机补充到500句
        remaining = [s for s in corpus if s not in final_corpus]
        needed = 500 - len(final_corpus)
        if needed > 0 and remaining:
            final_corpus.extend(random.sample(remaining, min(needed, len(remaining))))
        
        corpus = final_corpus[:500]
    
    # 最后检查：如果还有未覆盖的词，强制添加（替换一些句子）
    all_words_covered_final = set()
    for sentence in corpus:
        for word in words.keys():
            if word in sentence:
                all_words_covered_final.add(word)
    
    uncovered_words_final = set(words.keys()) - all_words_covered_final
    if uncovered_words_final:
        print(f"  发现 {len(uncovered_words_final)} 个未覆盖的词: {uncovered_words_final}")
        
        for word in uncovered_words_final:
            # 找到这个词所属的情绪类别
            found_category = None
            for category, word_list in emotion_categories:
                if word in word_list:
                    found_category = category
                    break
            
            if found_category:
                templates_list = short_templates.get(found_category, [])
                if templates_list:
                    template = random.choice(templates_list)
                    sentence = template.format(word=word)
                else:
                    sentence = f"我现在感到{word}"
                
                # 如果已经500句，替换一个已覆盖的词
                if len(corpus) >= 500:
                    # 找一个包含已覆盖词的句子替换
                    for idx, existing_sentence in enumerate(corpus):
                        # 检查这个句子是否包含多个已覆盖的词
                        words_in_sentence = [w for w in all_words_covered_final if w in existing_sentence]
                        if len(words_in_sentence) > 1:
                            corpus[idx] = sentence
                            all_words_covered_final.add(word)
                            break
                    else:
                        # 如果没找到，随机替换
                        idx = random.randint(0, len(corpus) - 1)
                        corpus[idx] = sentence
                        all_words_covered_final.add(word)
                else:
                    corpus.append(sentence)
                    all_words_covered_final.add(word)
    
    # 最终确保是500句
    if len(corpus) > 500:
        # 再次确保每个词至少有一句
        final_corpus = []
        words_covered_final = set()
        
        # 第一遍：为每个词选择一句
        for word in words.keys():
            for sentence in corpus:
                if word in sentence and word not in words_covered_final:
                    final_corpus.append(sentence)
                    words_covered_final.add(word)
                    break
        
        # 第二遍：随机补充到500句
        remaining = [s for s in corpus if s not in final_corpus]
        needed = 500 - len(final_corpus)
        if needed > 0 and remaining:
            final_corpus.extend(random.sample(remaining, min(needed, len(remaining))))
        
        corpus = final_corpus[:500]
    
    # 再次打乱
    random.shuffle(corpus)
    
    return corpus


def main():
    """主函数"""
    print("正在生成训练语料库...")
    corpus = generate_corpus()
    
    print(f"生成了 {len(corpus)} 句训练语料")
    
    # 统计信息
    lexicon = load_lexicon()
    all_words = set(lexicon['words'].keys())
    
    # 检查每句话是否包含词典词
    sentences_with_words = 0
    word_coverage = {word: 0 for word in all_words}
    
    for sentence in corpus:
        has_word = False
        for word in all_words:
            if word in sentence:
                word_coverage[word] += 1
                has_word = True
        if has_word:
            sentences_with_words += 1
    
    print(f"包含词典词的句子: {sentences_with_words}/{len(corpus)} ({sentences_with_words/len(corpus)*100:.1f}%)")
    
    # 统计句子长度分布
    short_sentences = sum(1 for s in corpus if len(s) < 20)
    medium_sentences = sum(1 for s in corpus if 20 <= len(s) < 40)
    long_sentences = sum(1 for s in corpus if len(s) >= 40)
    
    print(f"\n句子长度分布:")
    print(f"  短句 (<20字): {short_sentences} ({short_sentences/len(corpus)*100:.1f}%)")
    print(f"  中句 (20-40字): {medium_sentences} ({medium_sentences/len(corpus)*100:.1f}%)")
    print(f"  长句 (>=40字): {long_sentences} ({long_sentences/len(corpus)*100:.1f}%)")
    
    # 统计词典词覆盖情况
    covered_words = sum(1 for count in word_coverage.values() if count > 0)
    print(f"\n词典词覆盖: {covered_words}/{len(all_words)} ({covered_words/len(all_words)*100:.1f}%)")
    
    # 保存到文件
    output_path = Path(__file__).parent.parent / "data" / "training_corpus.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sentence in corpus:
            f.write(sentence + '\n')
    
    print(f"\n语料库已保存到: {output_path}")


if __name__ == "__main__":
    main()
