import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GCN
from deeprobust.graph.targeted_attack import Nettack
from deeprobust.graph.targeted_attack import FGA
from deeprobust.graph.global_attack import DICE
from deeprobust.graph.global_attack import Random
from deeprobust.graph.utils import *
from gad_dataset import Dataset
from tqdm import tqdm
import dgl
from scipy.sparse import csr_matrix
import os
import random
from utils import model_detector_dict
from sklearn.metrics import roc_auc_score, average_precision_score

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

def select_nodes(graph, idx_test, labels, target_gcn=None):
    target_gcn.eval()
    output = target_gcn(0, graph)
    log_probs = F.log_softmax(output, dim=1)
    margin_dict = {}
    for idx in idx_test:
        margin = classification_margin(log_probs[idx], labels[idx])
        if margin < 0: # only keep the nodes correctly classified
            continue
        margin_dict[idx] = margin
    sorted_margins = sorted(margin_dict.items(), key=lambda x:x[1], reverse=True)
    high = [x for x, y in sorted_margins[: 10]]
    low = [x for x, y in sorted_margins[-10: ]]
    # other = [x for x, y in sorted_margins[10: -10]]
    # other = np.random.choice(other, 20, replace=False).tolist()
    return high + low #+ other

def csr_to_tensor(csr):
    coo = csr.tocoo()
    indices = torch.tensor([coo.row, coo.col], dtype=torch.long)  # 索引需为long类型
    values = torch.tensor(coo.data, dtype=torch.float32)          # 数值转为float32
    shape = coo.shape                                             # 原始矩阵形状
    sparse_tensor = torch.sparse_coo_tensor(indices, values, shape)
    return sparse_tensor

def tensor_to_csr(tensor):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    dense_np = tensor.detach().numpy()
    csr_matrix = sp.csr_matrix(dense_np)
    return csr_matrix

def initialize_surrogate(dataset_name):
    """模型训练流程"""
    device = 'cuda'
    set_seed(3407)
    data = Dataset(dataset_name)
    data.split(False)
    graph = data.graph
    src, dst = graph.edges()
    num_nodes = graph.number_of_nodes()
    adj_data = np.ones(len(src))
    adj = csr_matrix((adj_data, (src.numpy(), dst.numpy())), shape=(num_nodes, num_nodes))

    features, labels, train_mask, val_mask, test_mask = graph.ndata["feature"], graph.ndata['label'] \
        , graph.ndata['train_mask'].bool(), graph.ndata['val_mask'].bool(), graph.ndata['test_mask'].bool()
    idx_train = np.arange(len(train_mask))[train_mask]
    idx_val = np.arange(len(val_mask))[val_mask]
    idx_test = np.arange(len(test_mask))[test_mask]
    gcn = GCN(nfeat=features.shape[1],
              nhid=16 if dataset_name == 'tsocial' else 64,
              nclass=labels.max().item() + 1,
              dropout=0, device=device)
    gcn = gcn.to(device)
    gcn.fit(features, adj, labels, idx_train, idx_val, verbose=True, patience=50)
    
    gcn.eval()
    output = gcn.predict()
    score = cal_score(output[idx_val], labels[idx_val])
    print("Overall val set results: AUROC= {:.4f} AUPRC= {:.4f} RecK= {:.4f}".format(score["AUROC"], score["AUPRC"], score["RecK"]))
    return adj, features, labels, idx_train, idx_val, idx_test, gcn

def cal_score(output, labels):
    probs = output.softmax(1)[:, 1]
    score = {}
    with torch.no_grad():
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        if torch.is_tensor(probs):
            probs = probs.cpu().numpy()
        score['AUROC'] = roc_auc_score(labels, probs)
        score['AUPRC'] = average_precision_score(labels, probs)
        labels = np.array(labels)
        k = labels.sum()
    score['RecK'] = sum(labels[probs.argsort()[-k:]]) / sum(labels)
    return score

def single_test(new_graph, target_node, labels, gcn=None):
    output = gcn(0, new_graph)
    probs = F.softmax(output, dim=1)
    current_post_conf = probs[target_node].max()
    acc_test = (output.argmax(1)[target_node] == labels[target_node])
    return acc_test.item(), current_post_conf

def target_test_evasion(target_name, dataset_name, attack_name):
    adj, features, labels, idx_train, idx_val, idx_test, surrogate = initialize_surrogate(dataset_name=dataset_name)
    features = tensor_to_csr(features)
    train_config = {'device': 'cuda', 'epochs': 200, 'patience': 50, 'metric': 'AUPRC', 'inductive': False, 'seed': 3407}
    target_config = {'model': target_name, 'lr': 0.01, 'drop_rate': 0, 'h_feats': 16 if dataset_name == 'tsocial' else 64}
    set_seed(train_config['seed'])
    data = Dataset(dataset_name)
    data.split(False)
    target_detector = model_detector_dict[target_name](0,0,0,None,train_config, target_config, data)
    test_score, logits = target_detector.train()
    probs = F.softmax(logits, dim=1)
    target = target_detector.best_model
    cnt = 0
    degrees = adj.sum(0).A1
    # adj = csr_to_tensor(adj)
    node_list = select_nodes(data.graph.to(train_config["device"]), idx_test, labels, target)
    num = len(node_list)
    pre_conf, post_conf, success = [], [], []
    
    print('=== [Evasion] Attacking %s nodes respectively ===' % num)
    for target_node in tqdm(node_list):
        current_pre_conf = probs[target_node].max()
        pre_conf.append(float(current_pre_conf.detach().cpu()))
        # n_perturbations = int(degrees[target_node])
        n_perturbations = int(degrees[target_node] * 0.2) 
        if attack_name == "nettack":
            model = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=False, device=train_config["device"])
            model = model.to(train_config["device"])
            model.attack(features, adj, labels, target_node, n_perturbations, verbose=False)
        elif attack_name == "fga":
            model = FGA(surrogate, nnodes=adj.shape[0], device=train_config["device"])
            model = model.to(train_config["device"])
            model.attack(features, adj, labels, idx_train, target_node, n_perturbations)
        modified_adj = model.modified_adj
        row_indices, col_indices = modified_adj.nonzero()
        new_graph = dgl.graph((row_indices, col_indices))
        new_graph.ndata["feature"] = csr_to_tensor(features)
        new_graph.ndata["label"] = labels
        new_graph.ndata["train_masks"] = data.graph.ndata['train_masks']
        new_graph.ndata["val_masks"] = data.graph.ndata['val_masks']
        new_graph.ndata["test_masks"] = data.graph.ndata['test_masks']
        acc, current_post_conf = single_test(new_graph.to(train_config["device"]), target_node, labels, gcn=target)
        post_conf.append(float(current_post_conf.detach().cpu())) 
        if acc == 0:
            cnt += 1
            success.append(True)
        else:
            success.append(False)
    print('misclassification rate : %s' % (cnt/num))
    return pre_conf, post_conf, success

def global_test_evasion(target_detector, data, attack_name, structral_attack_ratio=0.05):
    # train_config = {'device': 'cuda', 'epochs': 200, 'patience': 50, 'metric': 'AUPRC', 'inductive': False, 'seed': 3407}
    # target_config = {'model': target_name, 'lr': 0.01, 'drop_rate': 0, 'h_feats': 16 if dataset_name == 'tsocial' else 64}
    # set_seed(train_config['seed'])
    # data = Dataset(dataset_name)
    # data.split(False)
    # target_detector = model_detector_dict[target_name](train_config, target_config, data)
    # test_score, logits = target_detector.train()
    target = target_detector.best_model
    
    features, labels, train_masks, val_masks, test_masks = data.graph.ndata["feature"], data.graph.ndata["label"]\
        , data.graph.ndata["train_masks"], data.graph.ndata["val_masks"], data.graph.ndata["test_masks"]
    src, dst = data.graph.edges()
    num_nodes = data.graph.number_of_nodes()
    adj_data = np.ones(len(src))
    clean_csr_mat = csr_matrix((adj_data, (src.numpy(), dst.numpy())), shape=(num_nodes, num_nodes))
    n_perturbations = int(structral_attack_ratio * (clean_csr_mat.sum()//2))
    if attack_name == "random":
        attack_model = Random()
        attack_model.attack(clean_csr_mat, n_perturbations=n_perturbations)
    elif attack_name == "dice":
        attack_model = DICE()
        attack_model.attack(clean_csr_mat, labels, n_perturbations=n_perturbations)
    else:
        attack_model = None
    modified_adj = attack_model.modified_adj
    row_indices, col_indices = modified_adj.nonzero()
    new_graph = dgl.graph((row_indices, col_indices))
    new_graph.ndata["feature"] = features
    new_graph.ndata["label"] = labels
    new_graph.ndata["train_masks"] = train_masks
    new_graph.ndata["val_masks"] = val_masks
    new_graph.ndata["test_masks"] = test_masks

    logits = target(0, new_graph.to(target_detector.train_config['device']))
    # test_score = cal_score(logits[data.graph.ndata["test_mask"]], labels[data.graph.ndata["test_mask"]])
    # print("Overall val set results: AUROC= {:.4f} AUPRC= {:.4f} RecK= {:.4f}".format(test_score["AUROC"], test_score["AUPRC"], test_score["RecK"]))
    return logits

if __name__ == '__main__':
    # multi_test_poison()
    target_test_evasion(target_name="BWGNN", dataset_name="amazon", attack_name="fga")
    # new_graph = global_test_evasion(target_name="BWGNN", dataset_name="amazon", attack_name="dice", structral_attack_ratio=0.05)