### version 1

import torch
import numpy as np
import matplotlib.pyplot as plt
from gad_dataset import Dataset
import os
import random
from utils import model_detector_dict
import time
import seaborn as sns
from gad_attack import global_test_evasion, target_test_evasion
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# ================== 全局可视化参数配置 ==================
NEURIPS_STYLE = {
    'text.usetex': False,        # 禁用LaTeX以简化配置
    # 'font.family': 'sans-serif', # 现代学术字体
    # 'font.sans-serif': ['Arial'],
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.format': 'pdf',
    'lines.linewidth': 2,
    'axes.linewidth': 1.2,
    'grid.linestyle': '--',      # 虚线网格
    'grid.alpha': 0.3,
    'axes.facecolor': 'white',   # 坐标区域纯白
    'figure.facecolor': 'white', # 画布背景纯白
    'savefig.facecolor': 'white',# 保存背景色
    'savefig.edgecolor': 'white'
}

COLOR_PALETTE = {
    'abnormal': '#CC79A7',       # 浅橙色 (Pantone 137C)
    'normal': '#56B4E9',         # 学术蓝 (IBM Design)
    'expected': '#4D4D4D',       # 中性灰
    'hist_abnormal': '#DEEBF7',  # 浅蓝直方图
    'hist_normal': '#FBE5D6'     # 浅橙直方图
}

plt.style.use('seaborn-white')  # 使用白色基底样式
plt.rcParams.update(NEURIPS_STYLE)

# ================== 核心功能函数 ==================
def set_seed(seed=3407):
    """设置全局随机种子"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def compute_calibration(output, labels, test_mask, n_bins=20):
    """计算校准误差统计数据"""
    probs = torch.softmax(output[test_mask], dim=1).detach().cpu()
    confidences, preds = torch.max(probs, dim=1)
    accuracies = (preds == labels[test_mask]).float()
    bin_edges = torch.linspace(0, 1, n_bins+1)
    bin_indices = torch.bucketize(confidences, bin_edges[:-1])
    bin_acc = torch.zeros(n_bins)
    bin_conf = torch.zeros(n_bins)
    bin_counts = torch.zeros(n_bins)
    for i in range(n_bins):
        mask = (bin_indices == i)
        if mask.sum() > 0:
            bin_acc[i] = accuracies[mask].mean()
            bin_conf[i] = confidences[mask].mean()
            bin_counts[i] = mask.sum()
    return {
        'acc': bin_acc.numpy(),
        'conf': bin_conf.numpy(),
        'counts': bin_counts.numpy(),
        'edges': bin_edges.numpy()
    }

def plot_ece(calib_data, title, ax=None):
    """绘制校准误差图（样式优化版）"""
    ax = ax or plt.gca()
    centers = calib_data['edges'][:-1] + 0.5*(calib_data['edges'][1]-calib_data['edges'][0])
    
    # 主柱状图（浅橙色）
    ax.bar(centers, calib_data['acc'], 
           width=0.03, alpha=0.9, 
           color=COLOR_PALETTE['abnormal'],
           edgecolor='w',  # 白色边界增强对比
           linewidth=0.5,
           label='Model Accuracy')
    
    # 理想参考线（深灰色）
    ax.plot([0,1], [0,1], '--', 
            color=COLOR_PALETTE['expected'],
            linewidth=2.5, 
            label='Ideal Calibration')
    
    # 置信度分布直方图（蓝色）
    ax_hist = ax.inset_axes([0.58, 0.15, 0.35, 0.25], facecolor='white')
    ax_hist.bar(centers, calib_data['counts']/calib_data['counts'].sum(),
                width=0.03, alpha=0.9,
                color=COLOR_PALETTE['normal'],
                edgecolor='w',
                linewidth=0.5)
    ax_hist.grid(True, linestyle='--', alpha=0.3)
    ax_hist.set(xlabel='Confidence', ylabel='Density', xlim=[0,1])
    
    # 主图设置
    ax.set(xlim=[0,1], ylim=[0,1],
           xlabel='Prediction Confidence',
           ylabel='Empirical Accuracy',
           title=title)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper left', frameon=True, framealpha=0.9)
    return ax

def plot_confidence_distribution(conf_abnormal_misc, conf_abnormal_misc_dice, conf_normal_misc, conf_normal_misc_dice, title, ax=None):
    """置信度分布图（现代样式）"""
    ax = ax or plt.gca()
    # KDE曲线绘制
    sns.kdeplot(conf_abnormal_misc, color=COLOR_PALETTE['abnormal'], label='Clean-Ano.', linewidth=2.5, ax=ax)
    sns.kdeplot(conf_abnormal_misc_dice, color=COLOR_PALETTE['abnormal'], label='Global-Attack-Ano.', linestyle='-.', linewidth=2.5, ax=ax)
    sns.kdeplot(conf_normal_misc, color=COLOR_PALETTE['normal'], label='Clean-Norm.', linewidth=2.5, ax=ax)
    sns.kdeplot(conf_normal_misc_dice, color=COLOR_PALETTE['normal'], label='Global-Attack-Norm.', linestyle='-.', linewidth=2.5, ax=ax)
    # 统计标注
    ax.set(xlim=[0.2,1], xlabel='Prediction Confidence',
           ylabel='Density', title=title)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper left', framealpha=0.9)
    return ax

# ================== 实验流程函数 ==================
def training_pipeline(model_name, dataset_name, attack_type):
    """模型训练流程"""
    train_config = {'device': 'cuda', 'epochs': 200, 'patience': 50, 'metric': 'AUPRC', 'inductive': False, 'seed': 3407}
    model_config = {'model': model_name, 'lr': 0.01, 'drop_rate': 0, 'h_feats': 16 if dataset_name == 'tsocial' else 64}
    # 初始化实验环境
    set_seed(train_config['seed'])
    data = Dataset(dataset_name)
    data.split(False)
    # 模型训练
    detector = model_detector_dict[model_name](0,0,0,None,train_config, model_config, data)
    start_time = time.time()
    test_score, logits = detector.train()
    if attack_type != None:
        logits = global_test_evasion(target_detector=detector, data=data, attack_name=attack_type)
    return {'logits': logits, 'detector': detector, 'time_cost': time.time() - start_time}

def plot_target_attack_conf(pre_conf, post_conf, success, title, ax=None):
    ax = ax or plt.gca()
    # 主图：置信度变化分布
    sns.kdeplot(x=pre_conf, y=post_conf, cmap="Blues", fill=True, alpha=0.6,
                thresh=0.1, levels=5, ax=ax)
    scatter = sns.scatterplot(x=pre_conf, y=post_conf, hue=success,
                            palette={True:'#2166AC', False:'#B2182B'},
                            style=success, markers=['o', 'X'], s=80,
                            edgecolor='w', linewidth=0.8, ax=ax)
    # 参考线
    ax.plot([0,1], [0.5,0.5], '--', c='#666666', lw=2.5, alpha=0.8)
    ax.plot([0,1], [0,1], '-', c='#666666', lw=2.5, alpha=0.5)
    ax.text(0.95, 0.52, 'Decision Boundary', ha='right', va='bottom',
                color='#444444', fontsize=12)
    # 坐标轴设置（移除等比例限制）
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Pre-attack Confidence', ylabel='Post-attack Confidence', title=title)
    return ax

def generate_analysis_figures(results, model_name, dataset_name, pre_conf, post_conf, success):
    results_clean, results_global_attack = results
    """生成分析图表"""
    # 数据准备
    detector_clean = results_clean['detector']
    mask_clean = detector_clean.test_mask.detach().cpu()
    labels_clean = detector_clean.labels.detach().cpu()

    detector_global_attack = results_global_attack['detector']
    mask_global_attack = detector_global_attack.test_mask.detach().cpu()
    labels_global_attack = detector_global_attack.labels.detach().cpu()
    # 创建画布
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    # fig.suptitle(f"{model_name} on {dataset_name}", y=1.02, fontsize=18)
    # 绘制校准曲线
    calib_abnormal = compute_calibration(results_clean['logits'][labels_clean>0], labels_clean[labels_clean>0], mask_clean[labels_clean>0])
    plot_ece(calib_abnormal, '(a) Acc-Conf of Anomalous Nodes (Clean)', ax=axs[0][0])
    # 绘制校准曲线
    calib_normal = compute_calibration(results_clean['logits'][labels_clean==0], labels_clean[labels_clean==0], mask_clean[labels_clean==0])
    plot_ece(calib_normal, '(b) Acc-Conf of Normal Nodes (Clean)', ax=axs[0][1])
    # 保存结果
    # plt.tight_layout()
    # plt.savefig(f"figures/ECC_{model_name}_{dataset_name}_{attack_type}_analysis.pdf")
    # plt.close()
    # 绘制置信度分布
    # 创建画布
    # fig, axs = plt.subplots(1, 1, figsize=(8, 5))
    # fig.suptitle(f"{model_name} on {dataset_name}", y=1.02, fontsize=18)
    conf_abnormal_misc = get_misclassified_confidences(results_clean['logits'][labels_clean>0], labels_clean[labels_clean>0], mask_clean[labels_clean>0])
    conf_abnormal_misc_dice = get_misclassified_confidences(results_global_attack['logits'][labels_global_attack>0], 
                                labels_global_attack[labels_global_attack>0], mask_global_attack[labels_global_attack>0])
    # conf_abnormal_corr = get_corrlassified_confidences(results_clean['logits'][labels>0], labels[labels>0], mask[labels>0])
    conf_normal_misc = get_misclassified_confidences(results_clean['logits'][labels_clean==0], labels_clean[labels_clean==0], mask_clean[labels_clean==0])
    conf_normal_misc_dice = get_misclassified_confidences(results_global_attack['logits'][labels_global_attack==0], 
                                labels_global_attack[labels_global_attack==0], mask_global_attack[labels_global_attack==0])
    # conf_normal_corr = get_corrlassified_confidences(results_clean['logits'][labels==0], labels[labels==0], mask[labels==0])
    plot_confidence_distribution(conf_abnormal_misc, conf_abnormal_misc_dice, conf_normal_misc, conf_normal_misc_dice, 
                                 '(c) Confidence of Misclassified Nodes', ax=axs[1][0])
    # 保存结果 
    # plt.tight_layout()
    # plt.savefig(f"figures/Misc_Confidence_{model_name}_{dataset_name}_{attack_type}_analysis.pdf")
    plot_target_attack_conf(pre_conf, post_conf, success, '(d) Confidence under Target Attack', ax=axs[1][1])
    plt.tight_layout()
    plt.savefig(f"figures/Fig1_analysis.pdf")
    plt.close()

def get_misclassified_confidences(logits, labels, test_mask):
    """获取错分样本的置信度"""
    probs = torch.softmax(logits[test_mask], dim=1).detach().cpu()
    confidences, preds = torch.max(probs, dim=1)
    mis_mask = (preds != labels[test_mask])
    return confidences[mis_mask].cpu().numpy()

def get_corrlassified_confidences(logits, labels, test_mask):
    """获取正确分类样本的置信度"""
    probs = torch.softmax(logits[test_mask], dim=1).detach().cpu()
    confidences, preds = torch.max(probs, dim=1)
    corr_mask = (preds == labels[test_mask])
    return confidences[corr_mask].cpu().numpy()

# ================== 主程序 ==================
if __name__ == "__main__":
    MODEL = "BWGNN"
    DATASET = "amazon"
    ATTACK = None
    # 执行训练流程
    results_clean = training_pipeline(model_name=MODEL, dataset_name=DATASET, attack_type=None)
    results_global_attack = training_pipeline(model_name=MODEL, dataset_name=DATASET, attack_type="dice")
    pre_conf, post_conf, success = target_test_evasion(target_name=MODEL, dataset_name=DATASET, attack_name="fga")
    # 生成分析图表
    generate_analysis_figures(results=(results_clean, results_global_attack), model_name=MODEL, dataset_name=DATASET,
                              pre_conf=pre_conf, post_conf=post_conf, success=success)
