"""
可视化模块
用于可视化情感轨迹和维度分布
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import config

# 设置中文字体（macOS）
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


class EmotionVisualizer:
    """情感可视化器"""
    
    def __init__(self, output_dir=None):
        """
        初始化可视化器
        
        Args:
            output_dir: 输出目录，默认使用 config.OUTPUT_DIR
        """
        self.output_dir = Path(output_dir or config.OUTPUT_DIR)
        self.output_dir.mkdir(exist_ok=True)
    
    def plot_emotion_trajectory(self, emotion_coords: Dict[str, List[float]], 
                                tokens: Optional[List[str]] = None,
                                save_path: Optional[str] = None,
                                title: Optional[str] = None):
        """
        绘制情感轨迹图（时间线图）
        
        Args:
            emotion_coords: 情感坐标字典，如 {'valence': [...], 'arousal': [...]}
            tokens: Token 列表（可选，用于标注）
            save_path: 保存路径（可选）
            title: 图表标题（可选）
        """
        fig, axes = plt.subplots(len(emotion_coords), 1, figsize=(12, 4 * len(emotion_coords)))
        
        if len(emotion_coords) == 1:
            axes = [axes]
        
        for idx, (dim_name, values) in enumerate(emotion_coords.items()):
            ax = axes[idx]
            x = np.arange(len(values))
            ax.plot(x, values, marker='o', linewidth=2, markersize=4, label=dim_name)
            ax.set_xlabel('Token Position', fontsize=10)
            ax.set_ylabel(dim_name.capitalize(), fontsize=10)
            ax.set_title(f'{dim_name.capitalize()} Trajectory', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 如果有tokens，在x轴上标注所有tokens
            if tokens and len(tokens) == len(values):
                ax.set_xticks(range(len(tokens)))
                ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=9)
        
        if title:
            fig.suptitle(title, fontsize=14, y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            if not save_path.parent.exists():
                save_path.parent.mkdir(parents=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_valence_arousal_2d(self, valence: List[float], arousal: List[float],
                                tokens: Optional[List[str]] = None,
                                save_path: Optional[str] = None,
                                title: Optional[str] = None):
        """
        绘制 Valence-Arousal 二维散点图
        
        Args:
            valence: Valence 值列表
            arousal: Arousal 值列表
            tokens: Token 列表（可选）
            save_path: 保存路径（可选）
            title: 图表标题（可选）
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制轨迹线
        ax.plot(valence, arousal, 'o-', linewidth=2, markersize=6, alpha=0.7, label='Trajectory')
        
        # 标注起点和终点
        ax.plot(valence[0], arousal[0], 'go', markersize=10, label='Start')
        ax.plot(valence[-1], arousal[-1], 'ro', markersize=10, label='End')
        
        # 如果有tokens，标注关键点
        if tokens and len(tokens) == len(valence):
            # 只标注变化较大的点
            for i in range(0, len(tokens), max(1, len(tokens) // 10)):
                token_str = str(tokens[i])
                # 限制token显示长度
                if len(token_str) > 6:
                    token_str = token_str[:4] + '..'
                ax.annotate(token_str, (valence[i], arousal[i]), 
                           fontsize=8, alpha=0.7)
        
        ax.set_xlabel('Valence', fontsize=12)
        ax.set_ylabel('Arousal', fontsize=12)
        ax.set_title(title or 'Valence-Arousal 2D Trajectory', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 添加象限标注
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            if not save_path.parent.exists():
                save_path.parent.mkdir(parents=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_timeline(self, emotion_coords: Dict[str, List[float]],
                     tokens: Optional[List[str]] = None,
                     save_path: Optional[str] = None,
                     title: Optional[str] = None):
        """
        绘制时间线图（各维度随时间变化，合并在一张图上）
        
        Args:
            emotion_coords: 情感坐标字典
            tokens: Token 列表（可选）
            save_path: 保存路径（可选）
            title: 图表标题（可选）
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        x = np.arange(len(list(emotion_coords.values())[0]))
        
        for dim_name, values in emotion_coords.items():
            ax.plot(x, values, marker='o', linewidth=2, markersize=4, 
                   label=dim_name.capitalize(), alpha=0.8)
        
        ax.set_xlabel('Token Position', fontsize=12)
        ax.set_ylabel('Emotion Value', fontsize=12)
        ax.set_title(title or 'Emotion Timeline', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 如果有tokens，在x轴上标注
        if tokens:
            step = max(1, len(tokens) // 15)  # 最多显示15个标注
            tick_positions = list(range(0, len(tokens), step))
            tick_labels = []
            for i in tick_positions:
                if i < len(tokens):
                    token_str = str(tokens[i])
                    # 限制token显示长度
                    if len(token_str) > 8:
                        token_str = token_str[:6] + '..'
                    tick_labels.append(token_str)
                else:
                    tick_labels.append('')
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            if not save_path.parent.exists():
                save_path.parent.mkdir(parents=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_all(self, emotion_coords: Dict[str, List[float]],
                tokens: Optional[List[str]] = None,
                text: Optional[str] = None,
                save_prefix: Optional[str] = None):
        """
        绘制所有可视化图表
        
        Args:
            emotion_coords: 情感坐标字典
            tokens: Token 列表（可选）
            text: 原始文本（可选，用于标题）
            save_prefix: 保存文件前缀（可选）
        """
        title_suffix = f" - {text[:30]}..." if text and len(text) > 30 else (f" - {text}" if text else "")
        
        # 时间线图
        self.plot_timeline(
            emotion_coords, 
            tokens=tokens,
            title=f'Emotion Timeline{title_suffix}',
            save_path=f"{save_prefix}_timeline.png" if save_prefix else None
        )
        
        # 如果同时有valence和arousal，绘制2D图
        if 'valence' in emotion_coords and 'arousal' in emotion_coords:
            self.plot_valence_arousal_2d(
                emotion_coords['valence'],
                emotion_coords['arousal'],
                tokens=tokens,
                title=f'Valence-Arousal 2D{title_suffix}',
                save_path=f"{save_prefix}_2d.png" if save_prefix else None
            )
        
        # 单独的情感轨迹图
        self.plot_emotion_trajectory(
            emotion_coords,
            tokens=tokens,
            title=f'Emotion Trajectories{title_suffix}',
            save_path=f"{save_prefix}_trajectory.png" if save_prefix else None
        )
