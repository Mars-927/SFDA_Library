import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from utils.Project_Record import Project


def see_confidence_change_acc(now_epoch, groundtruth, clear_logic, confidence_change):
    confidence_increase_index = confidence_change["increase_inter_index"]
    confidence_decrease_index = confidence_change["decrease_inter_index"]
    confidence_other_index = confidence_change["other_inter_index"]

    if "see_confidence_change_acc" not in Project.global_save.keys():
        Project.global_save["see_confidence_change_acc"] = {
            "last_epoch": 0,
            "increase_counter": 0,
            "increase_pass": 0, 
            "other_counter": 0,
            "other_pass": 0,
            "decrease_counter": 0,
            "decrease_pass": 0,
        }
    
    if now_epoch != Project.global_save["see_confidence_change_acc"]["last_epoch"]:
        out_epoch = Project.global_save["see_confidence_change_acc"]['last_epoch']
        increase_counter = Project.global_save["see_confidence_change_acc"]['increase_counter'] + 0.0
        increase_pass = Project.global_save["see_confidence_change_acc"]['increase_pass'] + 0.0
        increase_acc = np.divide(increase_pass, increase_counter, out=np.zeros_like(increase_pass), where=increase_pass!=0)
        other_counter = Project.global_save["see_confidence_change_acc"]['other_counter'] + 0.0
        other_pass = Project.global_save["see_confidence_change_acc"]['other_pass'] + 0.0
        other_acc = np.divide(other_pass, other_counter, out=np.zeros_like(other_pass), where=other_pass!=0)
        decrease_counter = Project.global_save["see_confidence_change_acc"]['decrease_counter'] + 0.0
        decrease_pass = Project.global_save["see_confidence_change_acc"]['decrease_pass'] + 0.0
        decrease_acc = np.divide(decrease_pass, decrease_counter, out=np.zeros_like(decrease_pass), where=decrease_pass!=0)
        log_str = "epoch:{:>3} [I] {:>3.0f} / {:>3.0f}({:.4f}) [O] {:>3.0f} / {:>3.0f}({:.4f}) [D] {:>3.0f} / {:>3.0f}({:.4f})".format(out_epoch,increase_pass,increase_counter,increase_acc,other_pass,other_counter,other_acc,decrease_pass,decrease_counter,decrease_acc)
        Project.log(log_str, filename = "see_confidence_change_acc.txt",is_print=False)
        Project.global_save["see_confidence_change_acc"]['increase_counter'] = 0
        Project.global_save["see_confidence_change_acc"]['increase_pass'] = 0
        Project.global_save["see_confidence_change_acc"]['other_counter'] = 0
        Project.global_save["see_confidence_change_acc"]['other_pass'] = 0
        Project.global_save["see_confidence_change_acc"]['decrease_counter'] = 0
        Project.global_save["see_confidence_change_acc"]['decrease_pass'] = 0
        Project.global_save["see_confidence_change_acc"]['last_epoch'] = now_epoch

    predict = torch.argmax(clear_logic,dim=1)
    check = (groundtruth == predict)
    
    Project.global_save["see_confidence_change_acc"]['increase_counter'] += confidence_increase_index.shape[0]
    Project.global_save["see_confidence_change_acc"]['increase_pass'] += check[confidence_increase_index].sum().item()
    Project.global_save["see_confidence_change_acc"]['other_counter'] += confidence_other_index.shape[0]
    Project.global_save["see_confidence_change_acc"]['other_pass'] += check[confidence_other_index].sum().item()
    Project.global_save["see_confidence_change_acc"]['decrease_counter'] += confidence_decrease_index.shape[0]
    Project.global_save["see_confidence_change_acc"]['decrease_pass'] += check[confidence_decrease_index].sum().item()
    


def see_epoch_logic(PG, epoch):
    # 查看每一代的样本结果, 确定困难样本和简单样本
    # 输出信息: 第一列文件路径, 第二列~第N列当前图像是否分类正确, 正确样本使用 + 错误样本使用 _ 以qin
    now_epoch_predict = torch.argmax(PG.predict_line_storage,dim=1)
    ground_truth = PG.dataset_groundtruth
    path = PG.dataset_path
    check = (now_epoch_predict == ground_truth)
    if "see_epoch_logic" not in Project.global_save.keys():
        Project.global_save["see_epoch_logic"] = []
    Project.global_save["see_epoch_logic"].append(check.unsqueeze(dim=0))

    ### 打印
    folder = os.path.join(Project.root_path,f"see_epoch_pass.txt")
    check_tensor = Project.global_save["see_epoch_logic"]
    check_tensor = torch.concat(check_tensor,dim=0)
    check_tensor = (check_tensor.T).cpu().tolist()
    log_str = ""
    for index, item in enumerate(check_tensor):
        write_list = [f"{path[index]:<40}"]
        item_format = [ "+" if _ else "_" for _ in item ]
        write_list.extend(item_format)
        log_str += ' '.join(write_list) + "\n"
    with open(folder, 'w') as f:
            f.write(log_str)

def see_class_trend(PG, epoch):
    now_epoch_predict = torch.argmax(PG.predict_line_storage,dim=1)
    ground_truth = PG.dataset_groundtruth
    labels, counts_predict = torch.unique(now_epoch_predict,return_counts=True)
    _, counts_groundtruth = torch.unique(ground_truth,return_counts=True)
    labels_str = [f"{_:4d}" for _ in labels]
    counts_predict_str = [f"{_:4d}" for _ in counts_predict]
    counts_groundtruth_str = [f"{_:4d}" for _ in counts_groundtruth]
    log_str = f"====>{epoch}\nlabel   :{' '.join(labels_str)}\npredict :{' '.join(counts_predict_str)}\nground  :{' '.join(counts_groundtruth_str)}\n\n"
    folder = os.path.join(Project.root_path,f"see_epoch_balance.txt")
    with open(folder, 'a') as f:
            f.write(log_str)

def see_confusion_matrix(args, PG, writer, now_epoch):
    num_classes = args.class_num
    class_names = args.class_name
    figsize = [10, 8]
    preds = torch.argmax(PG.predict_line_storage, dim=1)
    labels = PG.dataset_groundtruth

    def get_confusion_matrix(preds, labels, num_classes):
        preds = torch.flatten(preds)
        labels = torch.flatten(labels)
        cmtx = confusion_matrix(labels.cpu(), preds.cpu(), labels=list(range(num_classes)))
        return cmtx
    
    def plot_confusion_matrix(cmtx, figsize, class_names):
        figure = plt.figure(figsize=figsize)
        plt.imshow(cmtx, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        threshold = cmtx.max() / 2.0
        for i, j in itertools.product(range(cmtx.shape[0]), range(cmtx.shape[1])):
            color = "white" if cmtx[i, j] > threshold else "black"
            plt.text(j,i,format(cmtx[i, j], ".2f") if cmtx[i, j] != 0 else ".",horizontalalignment="center",color=color,)
        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        return figure

    cmtx = get_confusion_matrix(preds, labels, num_classes)
    sub_cmtx = plot_confusion_matrix(cmtx, figsize, class_names)
    writer.add_figure(tag="Train Confusion Matrix", figure=sub_cmtx, global_step=now_epoch)

def see_pull_counter(pos_mask, now_epoch):
    if "see_pull_counter" not in Project.global_save.keys():
        Project.global_save["see_pull_counter"] = {
            "last_epoch": 0,
            "pull_counter": 0,
        }
    if now_epoch != Project.global_save["see_pull_counter"]["last_epoch"]:
        pull_counter = Project.global_save["see_pull_counter"]['decrease_pass']
        Project.global_save["see_pull_counter"]['pull_counter'] = 0
        Project.global_save["see_pull_counter"]['last_epoch'] = now_epoch
        folder = os.path.join(Project.root_path,f"see_pull_counter.txt")
        log_str = f"epoch:{now_epoch-1} pull counter:{pull_counter}\n"
        with open(folder, 'r') as f:
            f.write(log_str)
    Project.global_save["see_pull_counter"]['pull_counter'] += pos_mask.sum().item()
    

