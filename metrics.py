import torch
from sklearn.metrics import roc_auc_score


AUC_DATASETS = {'questions', 'minesweeper'}


def compute_metrics(logits, y, masks, dataset_name):
    use_auc = dataset_name.lower() in AUC_DATASETS
    results = {}
    if use_auc:
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        y_np  = y.cpu().numpy()
        for split, m in masks.items():
            m_cpu = m.cpu().numpy()
            results[split] = roc_auc_score(y_np[m_cpu], probs[m_cpu])
    else:
        preds = logits.argmax(dim=-1)
        for split, m in masks.items():
            results[split] = (preds[m] == y[m]).float().mean().item()
    return results
