import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import entropy
from collections import defaultdict

def cal_coverage(prediction_sets, val_labels):
    prediction_sets_abnormal = prediction_sets[np.where(val_labels == 1)[0], :]
    prediction_sets_normal = prediction_sets[np.where(val_labels == 0)[0], :]
    val_labels_abnormal = val_labels[np.where(val_labels==1)[0]]
    val_labels_normal = val_labels[np.where(val_labels==0)[0]]
    cov = prediction_sets[np.arange(prediction_sets.shape[0]), val_labels].mean()
    cov_abnormal = prediction_sets_abnormal[np.arange(prediction_sets_abnormal.shape[0]), val_labels_abnormal].mean()
    cov_normal = prediction_sets_normal[np.arange(prediction_sets_normal.shape[0]), val_labels_normal].mean()
    return [cov, cov_abnormal, cov_normal]

def cal_inefficiency(prediction_sets, val_labels):
    prediction_sets_abnormal = prediction_sets[np.where(val_labels == 1)[0], :]
    prediction_sets_normal = prediction_sets[np.where(val_labels == 0)[0], :]
    eff = np.sum(prediction_sets) / len(prediction_sets)
    eff_abnormal = np.sum(prediction_sets_abnormal) / len(prediction_sets_abnormal)
    eff_normal = np.sum(prediction_sets_normal) / len(prediction_sets_normal)
    return [eff, eff_abnormal, eff_normal]

def cal_ambiguity(prediction_sets, val_labels):
    prediction_sets_abnormal = prediction_sets[np.where(val_labels == 1)[0], :]
    prediction_sets_normal = prediction_sets[np.where(val_labels == 0)[0], :]
    prediction_sets_sum = prediction_sets.sum(axis=1)
    prediction_sets_abnormal_sum = prediction_sets_abnormal.sum(axis=1)
    prediction_sets_normal_sum = prediction_sets_normal.sum(axis=1)
    amb = (len(prediction_sets_sum[prediction_sets_sum==2]) + len(prediction_sets_sum[prediction_sets_sum==0])) / len(prediction_sets)
    amb_abnormal = (len(prediction_sets_abnormal_sum[prediction_sets_abnormal_sum==2]) + len(prediction_sets_abnormal_sum[prediction_sets_abnormal_sum==0])) / len(prediction_sets_abnormal)
    amb_normal = (len(prediction_sets_normal_sum[prediction_sets_normal_sum==2]) + len(prediction_sets_normal_sum[prediction_sets_normal_sum==0])) / len(prediction_sets_normal)
    return [amb, amb_abnormal, amb_normal]

def cal_singleton(prediction_sets, val_labels):
    prediction_sets_abnormal = prediction_sets[np.where(val_labels == 1)[0], :]
    prediction_sets_normal = prediction_sets[np.where(val_labels == 0)[0], :]
    val_labels_abnormal = val_labels[np.where(val_labels==1)[0]]
    val_labels_normal = val_labels[np.where(val_labels==0)[0]]
    prediction_sets_sum = prediction_sets.sum(axis=1)
    prediction_sets_abnormal_sum = prediction_sets_abnormal.sum(axis=1)
    prediction_sets_normal_sum = prediction_sets_normal.sum(axis=1)
    single = (prediction_sets[np.where(prediction_sets_sum == 1)[0], val_labels[np.where(prediction_sets_sum == 1)[0]]] == 1).sum() / len(prediction_sets)
    single_abnormal = (prediction_sets_abnormal[np.where(prediction_sets_abnormal_sum == 1)[0], val_labels_abnormal[np.where(prediction_sets_abnormal_sum == 1)[0]]] == 1).sum() / len(prediction_sets_abnormal)
    single_normal = (prediction_sets_normal[np.where(prediction_sets_normal_sum == 1)[0], val_labels_normal[np.where(prediction_sets_normal_sum == 1)[0]]] == 1).sum() / len(prediction_sets_normal)
    return [single, single_abnormal, single_normal]

def cal_fnr(prediction_sets, val_labels):
    prediction_sets_abnormal = prediction_sets[np.where(val_labels == 1)[0], :]
    prediction_sets_abnormal_sum = prediction_sets_abnormal.sum(axis=1)
    fnr = (prediction_sets_abnormal[np.where(prediction_sets_abnormal_sum == 1)[0], 0] == 1).sum() / len(prediction_sets_abnormal)
    return [fnr]

def cal_fpr(prediction_sets, val_labels):
    prediction_sets_normal = prediction_sets[np.where(val_labels == 0)[0], :]
    prediction_sets_normal_sum = prediction_sets_normal.sum(axis=1)
    fpr = (prediction_sets_normal[np.where(prediction_sets_normal_sum == 1)[0], 1] == 1).sum() / len(prediction_sets_normal)
    return [fpr]

func_list = [cal_coverage, cal_inefficiency, cal_ambiguity, cal_singleton, cal_fnr, cal_fpr]

def tps(cal_smx, val_smx, cal_labels, val_labels, n, alpha):
    metric_all_list = []
    cal_scores = 1 - cal_smx[np.arange(n), cal_labels]
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    qhat = np.quantile(cal_scores, q_level, interpolation='higher')
    prediction_sets = val_smx >= (1 - qhat)
    for func in func_list:
        for tmp in func(prediction_sets, val_labels):
            metric_all_list.append(tmp)
    return prediction_sets, metric_all_list

def aps(cal_smx, val_smx, cal_labels, val_labels, n, alpha):
    metric_all_list = []
    cal_pi = cal_smx.argsort(1)[:, ::-1]                        ### softmax descending order
    cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1).cumsum(axis=1)    ###
    cal_scores = np.take_along_axis(cal_srt, cal_pi.argsort(axis=1), axis=1)[range(n), cal_labels]
    qhat = np.quantile(cal_scores, np.ceil((n + 1) * (1 - alpha)) / n, interpolation="higher")
    val_pi = val_smx.argsort(1)[:, ::-1]
    val_srt = np.take_along_axis(val_smx, val_pi, axis=1).cumsum(axis=1)
    prediction_sets = np.take_along_axis(val_srt <= qhat, val_pi.argsort(axis=1), axis=1)
    for func in func_list:
        for tmp in func(prediction_sets, val_labels):
            metric_all_list.append(tmp)
    return prediction_sets, metric_all_list

def raps(cal_smx, val_smx, cal_labels, val_labels, n, alpha):
    metric_all_list = []
    lam_reg = 0.01
    k_reg = min(5, cal_smx.shape[1])
    disallow_zero_sets = False
    rand = True
    reg_vec = np.array(k_reg * [0, ] + (cal_smx.shape[1] - k_reg) * [lam_reg, ])[None, :]
    cal_pi = cal_smx.argsort(1)[:, ::-1]
    cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1)
    cal_srt_reg = cal_srt + reg_vec
    cal_L = np.where(cal_pi == cal_labels[:, None])[1]
    cal_scores = cal_srt_reg.cumsum(axis=1)[np.arange(n), cal_L] - np.random.rand(n) * cal_srt_reg[np.arange(n), cal_L]
    # Get the score quantile
    qhat = np.quantile(cal_scores, np.ceil((n + 1) * (1 - alpha)) / n, interpolation='higher')
    # Deploy
    n_val = val_smx.shape[0]
    val_pi = val_smx.argsort(1)[:, ::-1]
    val_srt = np.take_along_axis(val_smx, val_pi, axis=1)
    val_srt_reg = val_srt + reg_vec
    val_srt_reg_cumsum = val_srt_reg.cumsum(axis=1)
    indicators = (val_srt_reg.cumsum(axis=1) - np.random.rand(n_val,
                                                              1) * val_srt_reg) <= qhat if rand else val_srt_reg.cumsum(
        axis=1) - val_srt_reg <= qhat
    if disallow_zero_sets: indicators[:, 0] = True
    prediction_sets = np.take_along_axis(indicators, val_pi.argsort(axis=1), axis=1)
    for func in func_list:
        for tmp in func(prediction_sets, val_labels):
            metric_all_list.append(tmp)
    return prediction_sets, metric_all_list

"""
    Gets the value of lambda hat that controls the marginal risk for a monotone risk function.
    The calib loss table should be ordered from small loss to large loss
"""
def get_lhat(is_ano, calib_loss_table, lambdas, alpha, B=1):
    n = calib_loss_table.shape[0]
    rhat = calib_loss_table.mean(axis=0)
    if is_ano == 1:
        lhat_idx = max(np.argmax(((n/(n+1)) * rhat + B/(n+1) ) >= alpha) - 1, 0) # Can't be -1.
    else:
        lhat_idx = min(np.argmax(((n / (n + 1)) * rhat + B / (n + 1)) <= alpha) + 1, len(lambdas)-1)  # Can't be -1.
    return lambdas[lhat_idx]

# def crc(lambdas_example_table, cal_ano_miscover, cal_norma_miscover, val_ano_smx, val_norma_smx, val_ano_labels, val_norma_labels, alpha_risk):
def crc(lambdas_example_table, cal_ano_miscover, cal_norma_miscover, val_smx, val_labels, alpha_risk):
# def crc(lambdas_example_table, cal_ano_miscover, test_ano_miscover, cal_norma_miscover, test_norma_miscover, alpha_risk):
    metric_all_list = []
    lhat_ano_miscover = get_lhat(1, cal_ano_miscover, lambdas_example_table, alpha_risk)
    lhat_norma_miscover = get_lhat(0, cal_norma_miscover, lambdas_example_table, alpha_risk)
    prediction_sets = np.zeros_like(val_smx)
    # prediction_sets_ano = np.zeros_like(val_ano_smx)
    # prediction_sets_norma = np.zeros_like(val_norma_smx)
    # for tmp in range(len(val_ano_smx)):
    #     if val_ano_smx[tmp][1] >= lhat_ano_miscover:
    #         prediction_sets_ano[tmp] = np.array([1, 1])
    #     else:
    #         prediction_sets_ano[tmp] = np.array([1, 0])
    # for tmp in range(len(val_norma_smx)):
    #     if val_norma_smx[tmp][1] >= lhat_norma_miscover:
    #         prediction_sets_norma[tmp] = np.array([0, 1])
    #     else:
    #         prediction_sets_norma[tmp] = np.array([1, 1])
    # ano_miscover_loss = conditional_ano_miscoverage_loss(prediction_sets_ano, val_ano_labels)
    # norma_miscover_loss = conditional_norma_miscoverage_loss(prediction_sets_norma, val_norma_labels)
    # val_losses_lhat_ano_miscover_mean = ano_miscover_loss.mean()
    # val_losses_lhat_norma_miscover_mean = norma_miscover_loss.mean()
    for tmp in range(len(val_smx)):
        if val_smx[tmp][1] >= lhat_norma_miscover:
            if val_smx[tmp][1] >= lhat_ano_miscover:
                prediction_sets[tmp] = np.array([0, 1])
            else:
                prediction_sets[tmp] = np.array([0, 0])
        else:
            if val_smx[tmp][1] >= lhat_ano_miscover:
                prediction_sets[tmp] = np.array([1, 1])
            else:
                prediction_sets[tmp] = np.array([1, 0])

    # test_losses_lhat_ano_miscoverage = test_ano_miscover[:, np.argmax(lambdas_example_table == lhat_ano_miscover)]
    # test_losses_lhat_norma_miscoverage = test_norma_miscover[:, np.argmax(lambdas_example_table == lhat_norma_miscover)]

    fnr = cal_fnr(prediction_sets, val_labels)
    fpr = cal_fpr(prediction_sets, val_labels)
    val_sizes_lhat = np.sum(prediction_sets, axis=1)

    # val_losses_lhat_ano_miscover_mean = test_losses_lhat_ano_miscoverage.mean()
    # val_losses_lhat_norma_miscover_mean = test_losses_lhat_norma_miscoverage.mean()

    # val_sizes = val_sizes_lhat.mean()
    for func in func_list:
        for tmp in func(prediction_sets, val_labels):
            metric_all_list.append(tmp)
    return prediction_sets, metric_all_list, lhat_ano_miscover, lhat_norma_miscover, torch.tensor(val_sizes_lhat), fnr[0], fpr[0]
    # return None

def threshold(cal_smx, val_smx, cal_labels, val_labels, n, alpha):
    cal_pi = cal_smx.argsort(1)[:, ::-1]
    cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1).cumsum(axis=1)
    cal_scores = np.take_along_axis(cal_srt, cal_pi.argsort(axis=1), axis=1)[
        range(n), cal_labels
    ]

    val_pi = val_smx.argsort(1)[:, ::-1]
    val_srt = np.take_along_axis(val_smx, val_pi, axis=1).cumsum(axis=1)

    prediction_sets = np.take_along_axis(val_srt <= 1 - alpha, val_pi.argsort(axis=1), axis=1)
    prediction_sets[np.arange(prediction_sets.shape[0]), val_pi[:, 0]] = True

    cov = prediction_sets[np.arange(prediction_sets.shape[0]), val_labels].mean()
    eff = np.sum(prediction_sets) / len(prediction_sets)
    return prediction_sets, cov, eff

def get_example_loss_and_size_tables(scores, labels, lambdas_example_table, loss_func):
    lam_len = len(lambdas_example_table)
    loss_table = np.zeros((scores.shape[0], lam_len))
    for j in range(lam_len):
        est_labels = np.ones_like(scores)
        if labels[0][0] == 1:
            mask = np.where(scores[:,1] >= lambdas_example_table[j])[0]
            est_labels[mask] = np.array([0,1])
        else:
            mask = np.where(scores[:,1] < lambdas_example_table[j])[0]
            est_labels[mask] = np.array([1,0])
        loss_table[:,j] = loss_func(est_labels, labels)
    return loss_table

# def fpr_loss(est_labels, labels):
#     mask = (est_labels[np.arange(len(labels)), 0] == 0) & (est_labels[np.arange(len(labels)), 1] == 1)
#     fpr_loss = np.where(mask, 1, 0)
#     return fpr_loss
# def fnr_loss(est_labels, labels):
#     mask = (est_labels[np.arange(len(labels)), 1] == 0) & (est_labels[np.arange(len(labels)), 0] == 1)
#     fnr_loss = np.where(mask, 1, 0)
#     return fnr_loss
def conditional_ano_miscoverage_loss(est_labels, labels):
    mask = est_labels[np.arange(len(labels)), 1] == 0
    conditional_ano_miscoverage_loss = np.where(mask, 1, 0)
    return conditional_ano_miscoverage_loss

def conditional_norma_miscoverage_loss(est_labels, labels):
    mask = est_labels[np.arange(len(labels)), 0] == 0
    conditional_norma_miscoverage_loss = np.where(mask, 1, 0)
    return conditional_norma_miscoverage_loss

def run_conformal_classification(pred, data, n_base, alpha, alpha_risk, score='aps', return_prediction_sets=True):
    conformal_metric_name = ["Coverage", "Inefficiency", "Ambiguity", "Singleton", "FNR", "FPR"]
    conformal_group_name = ["", "Abnormal", "Normal"]
    logits = torch.nn.Softmax(dim = 1)(pred).detach().cpu().numpy()      ## all logits
    labels_all = data.graph.ndata["label"].detach().cpu().numpy()
    ###### 新加的
    # one_hot_array = np.zeros_like(logits)
    # one_hot_array[np.arange(len(labels_all)), labels_all] = 1
    # logits = one_hot_array * 0.8 + 0.1

    smx = logits[data.graph.ndata["calib_test_mask"]]          ## calib+test sets' logits
    labels = data.graph.ndata["label"][data.graph.ndata['calib_test_mask']].detach().cpu().numpy()      ## calib+test sets' labels
    metric_all_list = []
    for num in range((len(conformal_metric_name)-2)*len(conformal_group_name)+2):
        metric_all_list.append([])
    if return_prediction_sets:      ## record "prediction sets"/"test label sets"/"idx sets" in k splits
        pred_set_all = []
        test_labels_all = []
        idx_all = []

    num_lam = 1500
    lambdas_example_table = np.linspace(0, 1, num_lam)
    ano_index = np.where(labels == 1)[0]
    norma_index = np.where(labels == 0)[0]
    ano_scores = smx[ano_index]
    norma_scores = smx[norma_index]
    labels_onehot = np.eye(2)[labels]
    ano_labels_onehot = labels_onehot[ano_index]
    norma_labels_onehot = labels_onehot[norma_index]
    ano_example_miscover_loss_table = get_example_loss_and_size_tables(ano_scores, ano_labels_onehot, lambdas_example_table, conditional_ano_miscoverage_loss)
    norma_example_miscover_loss_table = get_example_loss_and_size_tables(norma_scores, norma_labels_onehot, lambdas_example_table, conditional_norma_miscoverage_loss)
    ano_num = len(ano_scores)
    norma_num = len(norma_scores)
    local_df_list = []
    df = pd.DataFrame(columns=["lambda_ano", "lambda_norma", "fnr", "fpr", "size", "alpha"])
    for k in range(1000):  ########## conformal prediction/risk control times
        idx_1 = np.array([1] * int(ano_num/2) + [0] * (ano_num - int(ano_num/2))) > 0
        np.random.seed(k)
        np.random.shuffle(idx_1)
        cal_ano_miscover, test_ano_miscover = ano_example_miscover_loss_table[idx_1, :], ano_example_miscover_loss_table[~idx_1, :]
        cal_ano_index, test_ano_index = ano_index[idx_1], ano_index[~idx_1]
        cal_ano_labels, test_ano_labels = ano_labels_onehot[idx_1], ano_labels_onehot[~idx_1]
        cal_ano_smx, test_ano_smx = ano_scores[idx_1], ano_scores[~idx_1]

        idx_2 = np.array([1] * int(norma_num/2) + [0] * (norma_num - int(norma_num/2))) > 0
        np.random.seed(k)
        np.random.shuffle(idx_2)
        cal_norma_miscover, test_norma_miscover = norma_example_miscover_loss_table[idx_2, :], norma_example_miscover_loss_table[~idx_2, :]
        cal_norma_index, test_norma_index = norma_index[idx_2], norma_index[~idx_2]
        cal_norma_labels, test_norma_labels = norma_labels_onehot[idx_2], norma_labels_onehot[~idx_2]
        cal_norma_smx, test_norma_smx = norma_scores[idx_2], norma_scores[~idx_2]

        cal_smx = smx[np.concatenate((cal_ano_index, cal_norma_index))]
        test_smx = smx[np.concatenate((test_ano_index, test_norma_index))]
        cal_labels = labels[np.concatenate((cal_ano_index, cal_norma_index))]
        test_labels = labels[np.concatenate((test_ano_index, test_norma_index))]
        n = cal_smx.shape[0]
        if score == 'tps':
            prediction_sets, metric_list = tps(cal_smx, test_smx, cal_labels, test_labels, n, alpha)
        elif score == 'aps':
            prediction_sets, metric_list = aps(cal_smx, test_smx, cal_labels, test_labels, n, alpha)
        elif score == 'raps':
            prediction_sets, metric_list = raps(cal_smx, test_smx, cal_labels, test_labels, n, alpha)
        elif score == 'crc':
            # prediction_sets,metric_list,lhat_ano_miscover,lhat_norma_miscover,sizes,fnr,fpr = \
            #     crc(lambdas_example_table, cal_ano_miscover, cal_norma_miscover, test_ano_smx, test_norma_smx, test_ano_labels, test_norma_labels, alpha_risk)
            prediction_sets,metric_list,lhat_ano_miscover,lhat_norma_miscover,sizes,fnr,fpr = \
                crc(lambdas_example_table, cal_ano_miscover, cal_norma_miscover, test_smx, test_labels, alpha_risk)
            # prediction_sets_all, metric_list_all,lhat_ano_miscover_all,lhat_norma_miscover_all,sizes_all,fnr_all,fpr_all = \
            #     crc(lambdas_example_table, cal_ano_miscover, cal_norma_miscover, logits, labels_all, alpha_risk)

            # 对每种边类型应用自定义函数
            # for etype in data.graph.canonical_etypes:
            #     u, v = data.graph.all_edges(etype=etype, order='eid')
            #     u_label = data.graph.nodes[u].data['label']
            #     v_label = data.graph.nodes[v].data['label']
            #     edge_type = (u_label == v_label).float() * 2 - 1
            #     data.graph.edges[etype].data['homophily'] = edge_type
            ##### 添加的内容
            # sizes_matrix = sizes_all.unsqueeze(0).expand(len(sizes_all), len(sizes_all))
            # sizes_diff = (sizes_matrix - sizes_matrix.T).abs()
            # adj_matrix = data.graph.adj().to_dense() - torch.eye(len(sizes_all))
            # non_adj_matrix = 1 - data.graph.adj().to_dense()
            # adj_diff = sizes_diff[adj_matrix==1]
            # non_adj_diff = sizes_diff[non_adj_matrix==1]
            # adj_zero_count, adj_one_count, adj_two_count = torch.sum(adj_diff.eq(0))/2.0, torch.sum(
            #     adj_diff.eq(1))/2.0, torch.sum(adj_diff.eq(2))/2.0
            # adj_total = adj_zero_count + adj_one_count + adj_two_count
            # print("adj size diff 0:{%.4f}, size diff 1:{%.4f}, size diff 2:{%.4f}"%(adj_zero_count/adj_total, adj_one_count/adj_total, adj_two_count/adj_total))
            ##### 添加的内容
            # for etype in data.graph.canonical_etypes:
            #     src, dst = data.graph.all_edges(etype=etype, order='eid')
            #     values = data.graph.edges[etype].data['homophily']
            #     # coo = (values, src, dst)
            #     weight_csr = torch.sparse_coo_tensor(indices=torch.stack([src,dst],dim=0), values=values, size=(len(sizes_all), len(sizes_all)),dtype=torch.float32, device=data.graph.device)
            #     weight_matrix = weight_csr.to_dense() - torch.eye(len(sizes_all))
            # homophily_matrix = (adj_matrix + weight_matrix) / 2.0
            # heterophily_matrix = (adj_matrix - weight_matrix) / 2.0
            # adj_homo_diff = sizes_diff[homophily_matrix == 1]
            # adj_homo_zero_count, adj_homo_one_count, adj_homo_two_count = torch.sum(adj_homo_diff.eq(0))/2.0, torch.sum(
            #     adj_homo_diff.eq(1))/2.0, torch.sum(adj_homo_diff.eq(2))/2.0
            # adj_homo_total = adj_homo_zero_count + adj_homo_one_count + adj_homo_two_count
            # print("adj homo size diff 0:{%.4f}, size diff 1:{%.4f}, size diff 2:{%.4f}"%(adj_homo_zero_count/adj_homo_total,
            #                                                                              adj_homo_one_count/adj_homo_total,
            #                                                                              adj_homo_two_count/adj_homo_total))
            # adj_hete_diff = sizes_diff[heterophily_matrix == 1]
            # adj_hete_zero_count, adj_hete_one_count, adj_hete_two_count = torch.sum(adj_hete_diff.eq(0))/2.0, torch.sum(
            #     adj_hete_diff.eq(1))/2.0, torch.sum(adj_hete_diff.eq(2))/2.0
            # adj_hete_total = adj_hete_zero_count + adj_hete_one_count + adj_hete_two_count
            # print("adj hete size diff 0:{%.4f}, size diff 1:{%.4f}, size diff 2:{%.4f}"%(adj_hete_zero_count/adj_hete_total,
            #                                                                              adj_hete_one_count/adj_hete_total,
            #                                                                              adj_hete_two_count/adj_hete_total))
            #
            # non_adj_zero_count, non_adj_one_count, non_adj_two_count = torch.sum(non_adj_diff.eq(0))/2.0, torch.sum(
            #     non_adj_diff.eq(1))/2.0, torch.sum(non_adj_diff.eq(2))/2.0
            # non_adj_total = non_adj_zero_count + non_adj_one_count + non_adj_two_count
            # print("non-adj size diff 0:{%.4f}, size diff 1:{%.4f}, size diff 2:{%.4f}"%(non_adj_zero_count/non_adj_total, non_adj_one_count/non_adj_total, non_adj_two_count/non_adj_total))

            # prediction_sets = crc(lambdas_example_table, cal_ano_miscover, test_ano_miscover, cal_norma_miscover, test_norma_miscover, alpha_risk)
        else: raise Exception
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

        if score == "crc":
            dict_local = {"lambda_ano": lhat_ano_miscover,
                          "lambda_norma": lhat_norma_miscover,
                          "fnr": fnr,
                          "fpr": fpr,
                          "size": [sizes],
                          "alpha": alpha_risk
                          }
            df_local = pd.DataFrame(dict_local, index=[k])
            local_df_list = local_df_list + [df_local]
    if score == "crc":
        df = pd.concat(local_df_list, axis=0, ignore_index=True)
        plot_histograms(df,alpha_risk)
        # plot_size_neigh_dis(data, labels_all, sizes)
        # print(1)

    if return_prediction_sets:
        return metric_all_list, pred_set_all, test_labels_all#, idx_all
    else:
        return None

def plot_histograms(df, alpha):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 3))

    fnr_minrisk = df['fnr'].min()
    fnr_maxrisk = df['fnr'].max()
    fnr_risk_bins = np.arange(fnr_minrisk, fnr_maxrisk, 0.02)

    fpr_minrisk = df['fpr'].min()
    fpr_maxrisk = df['fpr'].max()
    fpr_risk_bins = np.arange(fpr_minrisk, fpr_maxrisk, 0.02)

    size = torch.cat(df['size'].tolist(),dim=0).numpy()
    d = np.diff(np.unique(size)).min()
    lofb = size.min() - float(d) / 2
    rolb = size.max() + float(d) / 2
    size_bins = np.arange(lofb, rolb + d, d)

    axs[0].hist(df['fnr'], fnr_risk_bins, alpha=0.7, density=True)
    axs[0].set_xlabel('fnr')
    axs[0].locator_params(axis='x', nbins=4)
    axs[0].set_ylabel('density')
    axs[0].axvline(x=alpha, c='#999999', linestyle='--', alpha=0.7)

    axs[1].hist(df['fpr'], fpr_risk_bins, alpha=0.7, density=True)
    axs[1].set_xlabel('fpr')
    axs[1].locator_params(axis='x', nbins=4)
    axs[1].set_ylabel('density')
    axs[1].axvline(x=alpha, c='#999999', linestyle='--', alpha=0.7)

    axs[2].hist(size, size_bins, alpha=0.7, density=True)
    axs[2].set_xlabel('size')
    sns.despine(ax=axs[0], top=True, right=True)
    sns.despine(ax=axs[1], top=True, right=True)
    sns.despine(ax=axs[2], top=True, right=True)

    plt.tight_layout()
    os.makedirs('outputs/histograms/', exist_ok=True)
    plt.savefig((f'outputs/histograms/{alpha}_GAD_histograms').replace('.', '_') + '.pdf')
    # print(f"Average threshold: ", df["$\\hat{\\lambda}$"].mean())
    print(f"The mean and standard deviation of the fnr risk over {len(df)} trials are {df['fnr'].mean()} and {df['fnr'].std()} respectively.")
    print(f"The mean and standard deviation of the fpr risk over {len(df)} trials are {df['fpr'].mean()} and {df['fpr'].std()} respectively.")
    unique_values, counts = np.unique(size, return_counts=True)
    for value, count in zip(unique_values, counts):
        print(f"size value {value}: ratio {count/len(size)}")

def calculate_neighbor_entropy(node, adj, sizes_all):
    """计算指定节点邻域的熵值"""
    sizes_all = sizes_all.detach().tolist()
    neighbors = adj[node]
    if len(neighbors) == 0:
        return np.nan  # 处理孤立节点
    # 统计三类size的数量
    counts = np.zeros(3)
    for n in neighbors:
        counts[int(sizes_all[n])] += 1
    # 计算概率分布
    probabilities = counts / counts.sum()
    # 计算香农熵（以自然对数为底）
    return entropy(probabilities, base=np.e)

def plot_size_neigh_dis(data, labels_all, sizes_all):
    # 构建邻接表（考虑所有边类型和无向边）
    nodes = range(len(sizes_all))
    adj = defaultdict(set)
    etypes = data.graph.canonical_etypes
    for etype in etypes:
        edges = data.graph.edges(etype=etype)
        src = edges[0].numpy() if hasattr(edges[0], 'numpy') else edges[0]  # 兼容不同框架
        dst = edges[1].numpy() if hasattr(edges[1], 'numpy') else edges[1]
        for s, d in zip(src, dst):
            if s == d:
                continue
            adj[s].add(d)
            adj[d].add(s)
    # 转换为普通字典并清理空集合
    adj = {k: list(v) for k, v in adj.items() if len(v) > 0}
    # 计算所有节点的邻域熵值
    entropy_values = []
    for node in range(len(sizes_all)):
        if node not in adj:  # 孤立节点跳过
            continue
        e = calculate_neighbor_entropy(node, adj, sizes_all)
        if not np.isnan(e):
            entropy_values.append(e)
    # 创建数据框便于绘图
    df = pd.DataFrame({'Entropy': entropy_values})
    # 绘制概率密度分布
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x='Entropy', fill=True, alpha=0.6, linewidth=1.5, color='royalblue', label='Density')
    # 添加辅助元素
    plt.axvline(x=np.mean(entropy_values), color='red', linestyle='--', label=f'Mean: {np.mean(entropy_values):.2f}')
    plt.xlabel('Neighborhood Distribution Entropy', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.title('Distribution of Neighborhood Heterogeneity Entropy', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # 显示统计信息
    print(f"Entropy Statistics:\n"
          f"Mean: {np.mean(entropy_values):.2f}\n"
          f"Std : {np.std(entropy_values):.2f}\n"
          f"Min : {np.min(entropy_values):.2f}\n"
          f"Max : {np.max(entropy_values):.2f}")
    plt.show()
    print(1)

def plot_eff_diff(data, labels_all, sizes_all):
    eff_diff_homo_00 = defaultdict(int)
    eff_diff_homo_11 = defaultdict(int)
    eff_diff_hete = defaultdict(int)
    etypes = data.graph.canonical_etypes
    for etype in etypes:
        edges = data.graph.edges(etype=etype)
        src, dst = edges
        for i in range(len(src)):
            s, d = src[i], dst[i]
            if s == d: continue
            size_diff = str(int(abs(sizes_all[s] - sizes_all[d])))
            if labels_all[s] == labels_all[d] and labels_all[s] == 1:
                eff_diff_homo_11[size_diff] += 1
            elif labels_all[s] == labels_all[d] and labels_all[s] == 0:
                eff_diff_homo_00[size_diff] += 1
            else:
                eff_diff_hete[size_diff] += 1

    labels = ['0-0', '1-1', '0-1']
    size = [eff_diff_homo_00['0']+eff_diff_homo_00['1'], eff_diff_homo_11['0']+eff_diff_homo_11['1'], eff_diff_hete['0']+eff_diff_hete['1']]
    print("heterophily ratio:", float(size[2]/(size[0]+size[1]+size[2])))
    size_0 = [float(eff_diff_homo_00['0']/size[0]), float(eff_diff_homo_11['0']/size[1]), float(eff_diff_hete['0']/size[2])]
    size_1 = [float(eff_diff_homo_00['1']/size[0]), float(eff_diff_homo_11['1']/size[1]), float(eff_diff_hete['1']/size[2])]
    index = range(len(labels))
    fig, ax = plt.subplots()
    colors = ['#1f77b4', '#ff7f0e']
    bar_width = 0.35
    opacity = 0.8
    bar1 = ax.bar(index, size_0, bar_width, alpha=opacity, color=colors[0], label='Size Difference 0')
    bar2 = ax.bar([p + bar_width for p in index], size_1, bar_width, alpha=opacity, color=colors[1],
                  label='Size Difference 1')
    ax.set_xlabel('Group', fontsize=14)
    ax.set_ylabel('Number of Edges', fontsize=14)
    ax.set_title('Distribution of Size Difference in Groups', fontsize=16)
    ax.set_xticks([r + bar_width / 2 for r in range(len(size_0))])
    ax.set_xticklabels(labels, fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(loc='best', fontsize=12)
    plt.tight_layout()
    plt.show()