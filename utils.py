import random
import torch
import dgl
from models.detector import BaseGNNDetector, GHRNDetector, PCGNNDetector
from dgl.data.utils import load_graphs
import os
import numpy as np
import matplotlib.pyplot as plt

model_detector_dict = {
    # Classic Methods
    'MLP': BaseGNNDetector,

    # Standard GNNs
    'GCN': BaseGNNDetector,
    'GraphSAGE': BaseGNNDetector,
    'GAT': BaseGNNDetector,

    # Specialized GNNs
    'AMNet': BaseGNNDetector,
    'BWGNN': BaseGNNDetector,
    'GHRN': GHRNDetector,
    'PCGNN': PCGNNDetector,
    'NFGNN': BaseGNNDetector,
    'SSGNN': BaseGNNDetector
}

param_space = {}

param_space['MLP'] = {
    'h_feats': [16, 32, 64],
    'num_layers': [1, 2, 3, 4],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'activation': ['ReLU', 'LeakyReLU', 'Tanh']
}
param_space['GCN'] = {
    'h_feats': [16, 32, 64],
    'num_layers': [1, 2, 3],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'activation': ['ReLU', 'LeakyReLU', 'Tanh']
}
param_space['GraphSAGE'] = {
    'h_feats': [16, 32, 64],
    'num_layers': [1, 2, 3],
    'agg': ['mean', 'gcn', 'pool'],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'activation': ['ReLU', 'LeakyReLU', 'Tanh']
}
param_space['AMNet'] = {
    'h_feats': [16, 32, 64],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'num_layers': [1, 2, 3, 4],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'orders': [2, 3],
    'activation': ['ReLU', 'LeakyReLU', 'Tanh']
}
param_space['BWGNN'] = {
    'h_feats': [16, 32, 64],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'num_layers': [1, 2, 3, 4],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'mlp_layers': [1, 2],
    'activation': ['ReLU', 'LeakyReLU', 'Tanh'],
}
param_space['GHRN'] = {
    'h_feats': [16, 32, 64],
    'del_ratio': 10 ** np.linspace(-2, -1, 1000),
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'num_layers': [1, 2, 3, 4],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'mlp_layers': [1, 2],
}
param_space['GAT'] = {
    'h_feats': [16, 32],
    'num_heads': [1, 2, 4, 8],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'num_layers': [1, 2, 3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
}
param_space['PCGNN'] = {
    'h_feats': [16, 32, 64],
    'del_ratio': np.linspace(0.01, 0.8, 1000),
    'add_ratio': np.linspace(0.01, 0.8, 1000),
    'dist': ['euclidean', 'cosine'],
    # 'k': list(range(3, 10)),
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'num_layers': [1, 2, 3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
}

class Dataset:
    def __init__(self, name='tfinance', prefix='datasets/'):
        graph = load_graphs(prefix + name)[0][0]
        # graph = dgl.add_self_loop(graph)
        # src, dst = graph.edges()
        # graph.add_edges(dst, src)  # Add reverse edges
        self.name = name
        self.graph = graph
        self.num_classes = int(max(self.graph.ndata["label"]) + 1)
        self.num_features = int(self.graph.ndata["feature"].shape[1])

    def split(self, semi_supervised=True, trial_id=0):
        if semi_supervised:
            trial_id += 10
        self.graph.ndata['train_mask'] = self.graph.ndata['train_masks'][:,trial_id].bool()
        self.graph.ndata['val_mask'] = self.graph.ndata['val_masks'][:,trial_id].bool()
        self.graph.ndata['test_mask'] = self.graph.ndata['test_masks'][:,trial_id].bool()
        self.graph.ndata['calib_test_mask'] = self.graph.ndata['test_masks'][:,trial_id].bool()
        self.num_calib = int(sum(self.graph.ndata['calib_test_mask']) * 0.5)
        print(self.graph.ndata['train_mask'].sum(), self.graph.ndata['val_mask'].sum(), self.graph.ndata['test_mask'].sum())

def save_results(results, pickle_path):
    if not os.path.exists(pickle_path):
        os.makedirs(pickle_path, exist_ok=True)
    results.transpose().to_excel(pickle_path+'results.xlsx')
    print('save to file: {}'.format(pickle_path))

def sample_param(model, dataset, t=0):
    model_config = {'model': model, 'lr': 0.01, 'drop_rate': 0}
    if t == 0:
        return model_config
    for k, v in param_space[model].items():
        model_config[k] = random.choice(v)
    # Avoid OOM in Random Search
    if model in ['GAT', 'GATSep', 'GT'] and dataset in ['tfinance', 'dgraphfin', 'tsocial']:
        model_config['h_feats'] = 16
        model_config['num_heads'] = 2
    if dataset == 'tsocial':
        model_config['h_feats'] = 16
    if dataset in ['dgraphfin', 'tsocial']:
        if 'k' in model_config:
            model_config['k'] = min(5, model_config['k'])
        if 'num_cluster' in model_config:
            model_config['num_cluster'] = 2
        # if 'num_layers' in model_config:
        #     model_config['num_layers'] = min(2, model_config['num_layers'])
    return model_config

def plot_acc_calibration(idx_test, output, labels, n_bins, title):
    output = torch.softmax(output, dim=1)
    pred_label = torch.max(output[idx_test], 1)[1]
    p_value = torch.max(output[idx_test], 1)[0]
    ground_truth = labels[idx_test]
    confidence_all, confidence_acc = np.zeros(n_bins), np.zeros(n_bins)
    for index, value in enumerate(p_value):
        #value -= suboptimal_prob[index]
        interval = int(value / (1 / n_bins) -0.0001)
        confidence_all[interval] += 1
        if pred_label[index] == ground_truth[index]:
            confidence_acc[interval] += 1
    for index, value in enumerate(confidence_acc):
        if confidence_all[index] == 0:
            confidence_acc[index] = 0
        else:
            confidence_acc[index] /= confidence_all[index]
    start = np.around(1/n_bins/2, 3)
    step = np.around(1/n_bins, 3)
    plt.figure(figsize=(6, 4))
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams["font.weight"] = "bold"
    plt.bar(np.around(np.arange(start, 1.0, step), 3), confidence_acc,
            alpha=0.7, width=0.03, color='dodgerblue', label='Outputs')
    plt.bar(np.around(np.arange(start, 1.0, step), 3),
            np.around(np.arange(start, 1.0, step), 3), alpha=0.7, width=0.03, color='lightcoral', label='Expected')
    plt.plot([0,1], [0,1], ls='--',c='k')
    plt.xlabel('Confidence', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.tick_params(labelsize=13)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    #title = 'Uncal. - Cora - 20 - GCN'
    plt.title(title, fontsize=16, fontweight="bold")
    plt.legend(fontsize=14)
    plt.savefig('figures/' + title +'.png', format='png', dpi=300,
                pad_inches=0, bbox_inches = 'tight')
    plt.show()