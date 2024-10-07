import torch

def get_confidence_change(index, now_logic, prior_logic, confidence_change_tau = 0.025):
    # 计算置信度变化


    now_confidence = torch.max(now_logic,dim=1)[0]
    prior_confidence = torch.max(prior_logic,dim=1)[0]
    sub = now_confidence - prior_confidence
    confidence_increase_indices = torch.where(sub >= confidence_change_tau)[0]
    confidence_decrease_indices = torch.where(sub < -confidence_change_tau)[0]
    increase_index = set(confidence_increase_indices.cpu().tolist())
    decrease_index = set(confidence_decrease_indices.cpu().tolist())
    other_index = set(list(range(index.shape[0]))) - increase_index - decrease_index
    confidence_other_index = torch.tensor(list(other_index)).long().cuda()

    tmp = set(list(range(index.shape[0]))) - other_index
    confidence_fluctuate = torch.tensor(list(tmp)).long().cuda()

    return {
        "other_inter_index": confidence_other_index,
        "increase_inter_index": confidence_increase_indices,
        "decrease_inter_index": confidence_decrease_indices,
        "other_global_index": index[confidence_other_index],
        "increase_global_index": index[confidence_increase_indices],
        "decrease_global_index": index[confidence_decrease_indices],

        "confidence_fluctuate_inter": confidence_fluctuate,
        "confidence_fluctuate_global": index[confidence_fluctuate],
        "confidence_stable_inter": confidence_other_index,
        "confidence_stable_global": index[confidence_other_index],
    }

# 发现置信度变化比较稳定的样本可信度较高, 其他的样本可信度较差, 对于置信度发生严重变化的样本可能因为处于决策边界造成或者决策器对于这一部分样本不太平滑造成