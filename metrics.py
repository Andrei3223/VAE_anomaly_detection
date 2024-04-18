import torch
import numpy as np

from sklearn import metrics

@torch.no_grad()
def get_accuracy(loss, labels, thresh=5):
    '''
    loss: torch.Tensor batch_size x window_size x features
    labels: torch.Tensor | None
    '''
    # print(loss.shape)
    loss_w = loss.mean(dim=2)
    predictions = np.float32(loss_w.cpu() > thresh)
    if labels is None or len(labels) == 0:  # train
        labels = np.zeros(predictions.shape)
    else:
        labels = labels.cpu().numpy()
    return (predictions == labels).mean()


def get_fp_tp_rate(predict, actual):
    tn, fp, fn, tp = metrics.confusion_matrix(actual, predict, labels=[0, 1]).ravel()

    true_pos_rate = tp/(tp+fn)
    false_pos_rate = fp/(fp+tn)

    return false_pos_rate, true_pos_rate


def pak(scores, targets, thres, k=20):
    """
    :param scores: anomaly scores
    :param targets: target labels
    :param thres: anomaly threshold
    :param k: PA%K ratio, 0 equals to conventional point adjust and 100 equals to original predictions
    :return: point_adjusted predictions
    """
    scores = np.array(scores)
    thres = np.array(thres)

    predicts = scores > thres
    actuals = targets > 0.01

    one_start_idx = np.where(np.diff(actuals, prepend=0) == 1)[0]
    zero_start_idx = np.where(np.diff(actuals, prepend=0) == -1)[0]

    assert len(one_start_idx) == len(zero_start_idx) + 1 or len(one_start_idx) == len(zero_start_idx)

    if len(one_start_idx) == len(zero_start_idx) + 1:
        zero_start_idx = np.append(zero_start_idx, len(predicts))

    for i in range(len(one_start_idx)):
        if predicts[one_start_idx[i]:zero_start_idx[i]].sum() > k / 100 * (zero_start_idx[i] - one_start_idx[i]):
            predicts[one_start_idx[i]:zero_start_idx[i]] = 1

    return predicts


def pak_protocol(scores, labels, threshold, max_k=100):
    f1s = []
    ks = [k/100 for k in range(0, max_k + 1)]
    fprs = []
    tprs = []
    preds = []

    for k in range(max_k +1):
        adjusted_preds = pak(scores, labels, threshold, k=k)
        f1 = metrics.f1_score(labels, adjusted_preds)
        fpr, tpr = get_fp_tp_rate(adjusted_preds, labels)
        fprs.append(fpr)
        tprs.append(tpr)
        #print(f1)   
        #print(k)
        f1s.append(f1)
        preds.append(adjusted_preds)

    area_under_f1 = metrics.auc(ks, f1s)
    max_f1_k = max(f1s)
    k_max = f1s.index(max_f1_k)
    preds_for_max = preds[f1s.index(max_f1_k)]
    # import matplotlib.pyplot as plt
    # plt.cla()
    # plt.plot(ks, f1s)
    # plt.savefig('DiffusionAE/plots/PAK_PROTOCOL')
    print(f'AREA UNDER CURVE {area_under_f1}')
    
    return area_under_f1, max_f1_k, k_max, preds_for_max, fprs, tprs

