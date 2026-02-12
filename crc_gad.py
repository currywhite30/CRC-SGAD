## 功能：为GAD方法加入conformal risk control的后处理校准模块，并且通过训练后处理模块，
##      在保证假阴性和假阳性率的情况下尽可能地输出{0}或{1}。通过对比不同的基座模型，发现：
##      
## 要求:（1）bounded loss function L，随着Cλ(x)集合的增加，损失函数会减小，也就是说包含越多的答案、风险越小，λ能够控制集合大小的变化。
##          loss function例如假阴性率（实际为1但预测为0的比例）和假阳性（实际为0但预测为1的比例）损失，
##      （2）Cλ(x)集合为{k:f(X)k ≥ 1-λ}，即具有比较高的分数的类别的集合，λ是一个阈值。随着λ的增加，Cλ(x)集合的大小也会增加，集合越保守。
##      （3）conformal risk control会通过控制λ来控制漏掉的错误类，
## 过程：首先选择set-valued函数Cλ(x)，λ控制Cλ(x)的保守性；
##       其次选择一个上界为B的损失函数L(Cλ(x),y)，来衡量Cλ(x)的质量，并且要求L关于λ是非增的，即λ越大，L不变或减小
##       最后目标就是基于观察的校准集数据选择最优的λ，使得风险的期望值小于预定义的α
## 形式化定义：
##       给定n+1个可交换的非增、有界的随机函数值Li，由于L随着λ是非增的，所以一定有一个最大的λmax，使得所有的函数Li在λmax处的值小于等于α，
##       即风险被控制了，但是有可能过于保守，所以需要选一个合适的λ
## 解决方案：
##       在满足可交换性的条件下，使用前n个函数来选择阈值，
##       Rn^(λ)=(L1(λ)+...+Ln(λ))/n，给定任意的预期的风险等级上界α，定义
##       λ^ = inf{λ:n/(n+1)Rn^(λ)+B/(n+1)≤α}，
##       具体地，给定一个校准集合（有标签的），对于每一个样本，其Cλ(x)
##       假阴性风险的计算方式为
##
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import defaultdict
import random
import time
from gad_dataset import Dataset
from utils import model_detector_dict

def cal_coverage(prediction_sets, val_labels):
    prediction_sets_ano = prediction_sets[np.where(val_labels == 1)[0], :]
    prediction_sets_norm = prediction_sets[np.where(val_labels == 0)[0], :]
    val_labels_ano = val_labels[np.where(val_labels==1)[0]]
    val_labels_norm = val_labels[np.where(val_labels==0)[0]]
    cov = prediction_sets[np.arange(prediction_sets.shape[0]), val_labels].mean()
    cov_ano = prediction_sets_ano[np.arange(prediction_sets_ano.shape[0]), val_labels_ano].mean()
    cov_norm = prediction_sets_norm[np.arange(prediction_sets_norm.shape[0]), val_labels_norm].mean()
    return [cov, cov_ano, cov_norm]

def cal_inefficiency(prediction_sets, val_labels):
    prediction_sets_ano = prediction_sets[np.where(val_labels == 1)[0], :]
    prediction_sets_norm = prediction_sets[np.where(val_labels == 0)[0], :]
    eff = np.sum(prediction_sets) / len(prediction_sets)
    eff_ano = np.sum(prediction_sets_ano) / len(prediction_sets_ano)
    eff_norm = np.sum(prediction_sets_norm) / len(prediction_sets_norm)
    return [eff, eff_ano, eff_norm]

def cal_ambiguity(prediction_sets, val_labels):
    prediction_sets_ano = prediction_sets[np.where(val_labels == 1)[0], :]
    prediction_sets_norm = prediction_sets[np.where(val_labels == 0)[0], :]
    prediction_sets_sum = prediction_sets.sum(axis=1)
    prediction_sets_ano_sum = prediction_sets_ano.sum(axis=1)
    prediction_sets_norm_sum = prediction_sets_norm.sum(axis=1)
    amb = (len(prediction_sets_sum[prediction_sets_sum==2]) + len(prediction_sets_sum[prediction_sets_sum==0])) / len(prediction_sets)
    amb_ano = (len(prediction_sets_ano_sum[prediction_sets_ano_sum==2]) + len(prediction_sets_ano_sum[prediction_sets_ano_sum==0])) / len(prediction_sets_ano)
    amb_norm = (len(prediction_sets_norm_sum[prediction_sets_norm_sum==2]) + len(prediction_sets_norm_sum[prediction_sets_norm_sum==0])) / len(prediction_sets_norm)
    return [amb, amb_ano, amb_norm]

def cal_singleton(prediction_sets, val_labels):
    prediction_sets_ano = prediction_sets[np.where(val_labels == 1)[0], :]
    prediction_sets_norm = prediction_sets[np.where(val_labels == 0)[0], :]
    val_labels_ano = val_labels[np.where(val_labels==1)[0]]
    val_labels_norm = val_labels[np.where(val_labels==0)[0]]
    prediction_sets_sum = prediction_sets.sum(axis=1)
    prediction_sets_ano_sum = prediction_sets_ano.sum(axis=1)
    prediction_sets_norm_sum = prediction_sets_norm.sum(axis=1)
    single = (prediction_sets[np.where(prediction_sets_sum == 1)[0], val_labels[np.where(prediction_sets_sum == 1)[0]]] == 1).sum() / len(prediction_sets)
    single_abnormal = (prediction_sets_ano[np.where(prediction_sets_ano_sum == 1)[0], val_labels_ano[np.where(prediction_sets_ano_sum == 1)[0]]] == 1).sum() / len(prediction_sets_ano)
    single_normal = (prediction_sets_norm[np.where(prediction_sets_norm_sum == 1)[0], val_labels_norm[np.where(prediction_sets_norm_sum == 1)[0]]] == 1).sum() / len(prediction_sets_norm)
    return [single, single_abnormal, single_normal]

def cal_fnr(prediction_sets, val_labels):
    prediction_sets_ano = prediction_sets[np.where(val_labels == 1)[0], :]
    prediction_sets_ano_sum = prediction_sets_ano.sum(axis=1)
    fnr = (prediction_sets_ano[np.where(prediction_sets_ano_sum == 1)[0], 0] == 1).sum() / len(prediction_sets_ano)
    return [fnr]

def cal_fpr(prediction_sets, val_labels):
    prediction_sets_norm = prediction_sets[np.where(val_labels == 0)[0], :]
    prediction_sets_norm_sum = prediction_sets_norm.sum(axis=1)
    fpr = (prediction_sets_norm[np.where(prediction_sets_norm_sum == 1)[0], 1] == 1).sum() / len(prediction_sets_norm)
    return [fpr]

def conformal_classification_binary(logits, labels, data, method, alpha_fnr=0.1, alpha_fpr=0.1, return_prediction_sets=False):
    conformal_metric_name = ["Coverage", "Inefficiency", "Ambiguity", "Singleton", "FNR", "FPR"]
    conformal_group_name = ["", "Abnormal", "Normal"]
    probs = torch.nn.Softmax(dim=1)(logits).detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    probs_cal_test, labels_cal_test = probs[data.graph.ndata["test_mask"]], labels[data.graph.ndata["test_mask"]]
    metric_all_list = []
    for num in range((len(conformal_metric_name)-2)*len(conformal_group_name)+2):
        metric_all_list.append([])
    if return_prediction_sets:      ## record "prediction sets"/"test label sets"/"idx sets" in k splits
        pred_set_all = []
        test_labels_all = []
    num_lambdas = 1500
    lambdas_table = np.linspace(0, 1, num_lambdas)
    ano_index, norm_index = np.where(labels_cal_test == 1)[0], np.where(labels_cal_test == 0)[0]
    ano_probs, norm_probs = probs_cal_test[ano_index], probs_cal_test[norm_index]
    labels_cal_test_onehot = np.eye(2)[labels_cal_test]
    ano_labels_onehot, norm_labels_onehot = labels_cal_test_onehot[ano_index], labels_cal_test_onehot[norm_index] 
    ano_loss = get_lambdas_loss(ano_probs, ano_labels_onehot, lambdas_table, fnr_loss)
    norm_loss = get_lambdas_loss(norm_probs, norm_labels_onehot, lambdas_table, fpr_loss)
    ano_nums, norm_nums = len(ano_probs), len(norm_probs)
    local_df_list = []
    df = pd.DataFrame(columns=["lambda_ano", "lambda_norma", "fnr", "fpr", "size", "alpha"])
    for k in range(1000):
        idx_1 = np.array([1] * int(ano_nums/2) + [0] * (ano_nums - int(ano_nums/2))) > 0
        np.random.seed(k)
        np.random.shuffle(idx_1)
        cal_ano_loss, test_ano_loss = ano_loss[idx_1, :], ano_loss[~idx_1, :]
        cal_ano_idx, test_ano_idx = ano_index[idx_1, :], ano_index[~idx_1, :]

        idx_2 = np.array([1] * int(norm_nums/2) + [0] * (norm_nums - int(norm_nums/2))) > 0 
        np.random.seed(k)
        np.random.shuffle(idx_2)
        cal_norm_loss, test_norm_loss = norm_loss[idx_2, :], norm_loss[~idx_2, :]
        cal_norm_idx, test_norm_idx = norm_index[idx_2, :], norm_index[~idx_2, :]

        cal_probs = probs[np.concatenate((cal_ano_idx, cal_norm_idx))]
        test_probs = probs[np.concatenate((test_ano_idx, test_norm_idx))]
        cal_labels = labels[np.concatenate((cal_ano_idx, cal_norm_idx))]
        test_labels = labels[np.concatenate((test_ano_idx, test_norm_idx))]
        n = cal_probs.shape[0]
        if method == "crc":
            prediction_sets, metric_list, lambda_hat_ano, lambda_hat_norm, sizes, fnr, fpr = \
                dual_threshold_crc(lambdas_table, cal_ano_loss, cal_norm_loss, test_probs, test_labels, alpha_fnr, alpha_fpr)
            
        for num in range(len(conformal_metric_name)):
            if conformal_metric_name[num] != "FNR" and conformal_metric_name[num] != "FPR":
                metric_all_list[3 * num].append(metric_list[3 * num])
                metric_all_list[3 * num + 1].append(metric_list[3 * num + 1])
                metric_all_list[3 * num + 2].append(metric_list[3 * num + 2])
            else:
                metric_all_list[3 * num].append(metric_list[3 * num])
                metric_all_list[3 * num + 1].append(metric_list[3 * num + 1])
                break
        if return_prediction_sets:
            pred_set_all.append(prediction_sets)
            test_labels_all.append(test_labels)

        if method == "crc":
            dict_local = {"lambda_ano": lambda_hat_ano, "lambda_norma": lambda_hat_norm,
                          "fnr": fnr, "fpr": fpr, "size": [sizes], "alpha_fnr": alpha_fnr, "alpha_fpr": alpha_fpr}
            df_local = pd.DataFrame(dict_local, index=[k])
            local_df_list = local_df_list + [df_local]
    if method == "crc":
        df = pd.concat(local_df_list, axis=0, ignore_index=True)
        # plot_histograms(df,alpha_risk)
        # plot_eff_diff(data, labels_all, sizes_all)
        # print(1)
    if return_prediction_sets:
        return metric_all_list, pred_set_all, test_labels_all
    else:
        return None

func_list = [cal_coverage, cal_inefficiency, cal_ambiguity, cal_singleton, cal_fnr, cal_fpr]

def dual_threshold_crc(lambdas_table, cal_ano_loss, cal_norm_loss, val_probs, val_labels, alpha_fnr, alpha_fpr):
    metric_all_list = []
    lambda_hat_ano = get_lambdas_hat(1, cal_ano_loss, lambdas_table, alpha_fnr)
    lambda_hat_norm = get_lambdas_hat(0, cal_norm_loss, lambdas_table, alpha_fpr)
    prediction_sets = np.zeros_like(val_probs)
    for tmp in range(len(val_probs)):
        if val_probs[tmp][1] >= lambda_hat_norm:
            if val_probs[tmp][1] >= lambda_hat_ano:
                prediction_sets[tmp] = np.array([0,1])
            else:
                prediction_sets[tmp] = np.array([0,0])
        else:
            if val_probs[tmp][1] >= lambda_hat_ano:
                prediction_sets[tmp] = np.array([1,1])
            else:
                prediction_sets[tmp] = np.array([1,0])
    fnr = cal_fnr(prediction_sets, val_labels)
    fpr = cal_fpr(prediction_sets, val_labels)
    val_sizes = np.sum(prediction_sets, axis=1)
    for func in func_list:
        for tmp in func(prediction_sets, val_labels):
            metric_all_list.append(tmp)
    return prediction_sets, metric_all_list, lambda_hat_ano, lambda_hat_norm, torch.tensor(val_sizes), fnr[0], fpr[0]

def get_lambdas_hat(is_ano, calib_loss_table, lambdas, alpha, B=1):
    ### 分别计算0类和1类的最优阈值
    n = calib_loss_table.shape[0]
    rhat = calib_loss_table.mean(axis=0)
    if is_ano == 1:
        lhat_idx = max(np.argmax(((n/(n+1)) * rhat + B/(n+1) ) >= alpha) - 1, 0) # Can't be -1.
    else:
        lhat_idx = min(np.argmax(((n / (n + 1)) * rhat + B / (n + 1)) <= alpha) + 1, len(lambdas)-1)  # Can't be -1.
    return lambdas[lhat_idx]

def get_lambdas_loss(probs, labels, lambdas_table, loss_func):
    lam_len = len(lambdas_table)
    ### 每个样本随着不同的λ阈值的loss值，由于正常样本和异常样本的损失计算方式不一样，比如对于正常样本，应该是计算假阳性损失。所以需要按照类别分别控制两种loss
    ###     那么计算set-valued函数的方式应不应该一样呢？
    ###     已知set-valued函数的要求是λ越大，集合越保守，如果
    loss_table = np.zeros((probs.shape[0], lam_len))
    for j in range(lam_len):
        est_labels = np.ones_like(probs)            
        if labels[0][0] == 1:   ## 对应normal的样本
            mask = np.where(probs[:,1] <= lambdas_table[j])[0]
            est_labels[mask] = np.array([0,1])      ## normal样本的set_valued函数的定义为：如果预测为异常的概率大于等于λ，则预测集合为{1}，否则为{0,1}
        else:                   ## 对应anomaly的样本
            mask = np.where(probs[:,1] <= lambdas_table[j])[0]
            est_labels[mask] = np.array([1,0])      ## anomaly样本的set_valued函数的定义为：如果预测为异常的概率小于λ，则预测集合为{0}，否则为{0,1}
        loss_table[:,j] = loss_func(est_labels, labels)
    return loss_table

def fnr_loss(est_labels, labels):
    ### 假阴性损失、漏判
    mask = est_labels[np.arange(len(labels)), 1] == 0
    fnr_loss = np.where(mask, 1, 0)
    return fnr_loss

def fpr_loss(est_labels, labels):
    ### 假阳性损失、误判
    mask = est_labels[np.arange(len(labels)), 0] == 0
    fpr_loss = np.where(mask, 1, 0)
    return fpr_loss

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

if __name__ == "__main__":
    model_name, dataset_name = "BWGNN", "amazon"
    train_config = {'device': 'cuda', 'epochs': 200, 'patience': 50, 'metric': 'AUPRC', 'inductive': False, 'seed': 3407}
    model_config = {'model': model_name, 'lr': 0.01, 'drop_rate': 0, 'h_feats': 16 if dataset_name == 'tsocial' else 64}
    # 初始化实验环境
    set_seed(train_config['seed'])
    data = Dataset(dataset_name)
    data.split(False)
    # 模型训练
    detector = model_detector_dict[model_name](train_config, model_config, data)
    start_time = time.time()
    test_score, logits = detector.train()
    n_calib = data.num_calib
    method = "crc"
    alpha_fnr = 0.1
    alpha_fpr = 0.1
    labels = data.graph.ndata["label"]
    metric_all, pred_set_all, val_labels_all = conformal_classification_binary(logits, labels, data, method, alpha_fnr, 
                                                                               alpha_fpr, return_prediction_sets=True)