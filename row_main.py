import argparse
from utils import model_detector_dict, Dataset, save_results, plot_acc_calibration
from row_attack import add_perturbation
from row_conformal import run_conformal_classification
from models.gnn import ConfGNN
import time
import pandas as pd
import os
import torch
import warnings
import random
import copy
import numpy as np
warnings.filterwarnings("ignore")
seed_list = list(range(3407, 10000, 10))
def set_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    #####
    # 代码的工作流: 原始的GAD + conformal prediction/crc or
    #             攻击的GAD + conformal prediction/crc or
    #             不同测试集划分的GAD(具有分布偏移) + conformal prediction/crc
    #####
    parser = argparse.ArgumentParser(description='HyperParameters')
    ### GAD's parameters
    parser.add_argument('--work_type', type=str, default='normal', choices=['normal', 'adversarial', 'sds'], help='type of different work flow')
    parser.add_argument('--trials', type=int, default=1, help='number of experiments')
    parser.add_argument('--models', type=str, default='AMNet', choices=['AMNet', 'NFGNN','BWGNN','GCN','GAT','GraphSAGE','PCGNN','GHRN'])
    parser.add_argument('--datasets', type=str, default='amazon', choices=['weibo', 'questions', 'amazon', 'reddit', 'yelp'])
    parser.add_argument('--attack_type', type=str, default='none', choices=['random', 'dice', 'none'])
    parser.add_argument('--struct_attack_ratios', type=float, default=0.1)
    parser.add_argument('--semi_supervised', type=int, default=0, help='0:40%   1:5%')
    parser.add_argument('--inductive', type=int, default=0, help='0: transductive    1: inductive')
    ### plot parameters
    parser.add_argument('--n_bins', type=int, default=20, help='')
    ### conformal prediction's parameters
    parser.add_argument('--alpha', type=float, default=0.1, help='miscoverage rate')
    parser.add_argument('--alpha_risk', type=float, default=0.1, help='risk rate')
    ### conformal training's parameters
    parser.add_argument('--is_conformal_training', type=int, default=1, choices=[0,1], help='1:conformal training  0:normal training')
    parser.add_argument('--size_loss_weight', type=float, default=1, help='weight of size loss')
    args = parser.parse_args()
    alpha = args.alpha
    alpha_risk = args.alpha_risk
    ### all available datasets' names / models' names / structral attack ratios / metrics
    metric_name_list = [
        'Time', 'AUROC mean', 'AUROC std', 'AUPRC mean', 'AUPRC std', 'RecK mean', 'RecK std',
        *[f"{m} {g} {s} {me}" if m not in ["FNR", "FPR"] else f"{m} {s} {me}"
          for s in ["TPS", "APS", "RAPS", "CRC"]
          for m in ["Coverage", "Inefficiency", "Ambiguity", "Singleton", "FNR", "FPR"]
          for g in ["", "Abnormal", "Normal"]
          for me in ["mean", "std"]
          if not (m in ["FNR", "FPR"] and g)]
    ]
    ### get datasets' names and models' names and structral attack ratios during this run
    # Struct attack ratios: {struct_attack_ratios}\n
    dataset_name, struct_attack_ratios, model_name, attack_type = args.datasets, args.struct_attack_ratios, args.models, args.attack_type
    print(f'Evaluated Datasets: {dataset_name}\nEvaluated Baselines: {model_name}')
    ### create information in result table
    if args.work_type == 'adversarial':
        columns = ['name'] + [f"{dataset_name}-{attack_type}-{struct_attack_ratios}-{metric}" for metric in metric_name_list]
    else:
        columns = ['name'] + [f"{dataset_name}-{metric}" for metric in metric_name_list]
    results = pd.DataFrame(columns=columns)
    if args.is_conformal_training:
        conf_metric_all_list = [[] for _ in range(len(metric_name_list) // 2)]
    ### graph anomaly detection's training loop (for every [model, dataset and attack ratio])
    model_result = {'name': model_name}
    time_cost, conf_time_cost = 0, 0
    data = Dataset(dataset_name)
    if args.work_type == 'adversarial':
        attacked_data = Dataset(dataset_name)
        new_graph = add_perturbation(data, attack_type, struct_attack_ratios)
        data.graph = new_graph
    train_config = {'device': 'cuda', 'epochs': 200, 'patience': 50, 'metric': 'AUPRC', 'inductive': bool(args.inductive)}
    model_config = {'model': model_name, 'lr': 0.01, 'drop_rate': 0}
    metric_all_list = [[] for _ in range(len(metric_name_list) // 2)]
    for t in range(args.trials):
        torch.cuda.empty_cache()
        print(f"Dataset {dataset_name}, Model {model_name}, struct attack ratio {struct_attack_ratios}, Trial {t}")
        data.split(args.semi_supervised, t)
        set_seed(seed_list[t])
        # train_config['seed'] = seed
        ###### original training process
        detector = model_detector_dict[model_name](0, 0, 0, None, train_config, model_config, data)
        st = time.time()
        test_score, logits = detector.train()
        [metric_all_list[i].append(test_score[metric]) for i, metric in enumerate(['AUROC', 'AUPRC', 'RecK'])]
        ed = time.time()
        time_cost += ed - st
        ###### plot acc-confidence figure
        title_abnormal = 'Abnormal - Uncal. - %s - %s - %s - %f - %d' % (model_name, dataset_name, attack_type, struct_attack_ratios, t)
        title_normal = 'Normal - Uncal. - %s - %s - %s - %f - %d' % (model_name, dataset_name, attack_type, struct_attack_ratios, t)
        logits_abnormal = logits[detector.labels>0]
        logits_normal = logits[detector.labels==0]
        test_mask_abnormal = detector.test_mask[detector.labels>0]
        test_mask_normal = detector.test_mask[detector.labels==0]
        labels_abnormal = detector.labels[detector.labels>0]
        labels_normal = detector.labels[detector.labels==0]
        # plot_acc_calibration(test_mask_abnormal, logits_abnormal, labels_abnormal, args.n_bins, title_abnormal)
        # plot_acc_calibration(test_mask_normal, logits_normal, labels_normal, args.n_bins, title_normal)
        ###### original conformal prediction / conformal risk control
        n_calib = data.num_calib
        scores = {'tps': ('tps_metric_all', 'pred_set_all_tps', 'val_labels_all_tps'),
                  'aps': ('aps_metric_all', 'pred_set_all_aps', 'val_labels_all_aps'),
                  'raps': ('raps_metric_all', 'pred_set_all_raps', 'val_labels_all_raps'),
                  'crc': ('crc_metric_all', 'pred_set_all_crc', 'val_labels_all_crc')}
        results_dict = {}
        for score, (metric_var, pred_set_var, val_labels_var) in scores.items():
            metric_all, pred_set_all, val_labels_all = run_conformal_classification(logits, data, n_calib, alpha, alpha_risk,
                                                                                    score=score, return_prediction_sets=True)
            results_dict[score] = (metric_all, pred_set_all, val_labels_all)
        for i, score in enumerate(scores.keys()):
            metric_all = results_dict[score][0]
            for num in range(len(metric_all)):
                metric_all_list[3 + len(metric_all) * i + num] += metric_all[num]
        ###### conformal training process
        if args.is_conformal_training == 1:
            backbone, model_to_correct = "SSGNN", copy.deepcopy(detector.model)
            conf_model = ConfGNN(model=model_to_correct, backbone=backbone, h_feats=64, mlp_layers=2).to('cuda')
            conf_train_config = {'device': 'cuda', 'epochs': 200, 'patience': 50, 'metric': 'AUPRC',
                                 'inductive': bool(args.inductive)}
            conf_model_config = {'model': backbone, 'lr': 0.01, 'drop_rate': 0}
            # conf_model_config = {'model': backbone, 'lr': 0.005, 'prop_lr': 0.01, 'weight_decay': 0.0005,
            #                      'prop_wd': 0.0005, 'drop_rate': 0.1}
            conf_detector = model_detector_dict[backbone](backbone, args.is_conformal_training, args.size_loss_weight,
                                                          conf_model, conf_train_config, conf_model_config, data)
            conf_st = time.time()
            conf_test_score, conf_logits = conf_detector.train()
            [conf_metric_all_list[i].append(conf_test_score[metric]) for i, metric in enumerate(['AUROC', 'AUPRC', 'RecK'])]
            conf_ed = time.time()
            conf_time_cost += conf_ed - conf_st
            n_calib = data.num_calib
            conf_results_dict = {}
            for score, (metric_var, pred_set_var, val_labels_var) in scores.items():
                conf_metric_all, conf_pred_set_all, conf_val_labels_all = run_conformal_classification(conf_logits, data, n_calib, alpha,
                                                                        alpha_risk, score=score, return_prediction_sets=True)
                conf_results_dict[score] = (conf_metric_all, conf_pred_set_all, conf_val_labels_all)
            for i, score in enumerate(scores.keys()):
                conf_metric_all = conf_results_dict[score][0]
                for num in range(len(conf_metric_all)):
                    conf_metric_all_list[3 + len(conf_metric_all) * i + num] += conf_metric_all[num]
    prefix = f'{dataset_name}-{attack_type}-{struct_attack_ratios}'
    if args.is_conformal_training == 1:
        conf_results = pd.DataFrame(columns=columns)
        conf_model_result = {'name': model_name}
        del conf_detector
        conf_model_result[prefix + "-Time"] = conf_time_cost / args.trials
        for num in range(len(conf_metric_all_list)):
            conf_model_result[prefix + "-" + metric_name_list[num * 2 + 1]] = np.mean(conf_metric_all_list[num])
            conf_model_result[prefix + "-" + metric_name_list[num * 2 + 1 + 1]] = np.std(conf_metric_all_list[num])
        conf_pickle_path = 'results_{}/{}/{}/{}/'.format(attack_type, "conf_" + model_name, dataset_name, str(struct_attack_ratios))
        conf_model_result = pd.DataFrame(conf_model_result, index=[0])
        conf_results = pd.concat([conf_results, conf_model_result])
        save_results(conf_results, conf_pickle_path)
    del detector, data
    model_result[prefix + "-Time"] = time_cost / args.trials
    for num in range(len(metric_all_list)):
        model_result[prefix + "-" + metric_name_list[num * 2 + 1]] = np.mean(metric_all_list[num])
        model_result[prefix + "-" + metric_name_list[num * 2 + 1 + 1]] = np.std(metric_all_list[num])
    pickle_path = 'results_{}/{}/{}/{}/'.format(attack_type, model_name, dataset_name, str(struct_attack_ratios))
    model_result = pd.DataFrame(model_result, index=[0])
    results = pd.concat([results, model_result])
    save_results(results, pickle_path)