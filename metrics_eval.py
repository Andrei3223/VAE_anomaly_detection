import torch
import numpy as np

from sklearn import metrics

from tqdm import tqdm


@torch.no_grad()
def get_accuracy(loss, labels, thresh=5):
    '''
    loss: torch.Tensor batch_size x window_size x features
    labels: torch.Tensor | None
    '''
    # print(loss.shape)
    loss_w = loss.mean(axis=2)
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

    # print(scores.shape, targets.shape, thres)

    predicts = scores > thres
    actuals = targets > 0.01

    # print(np.diff(actuals, prepend=0))

    one_start_idx = np.where(np.diff(actuals, prepend=0) == 1)[0]
    zero_start_idx = np.where(np.diff(actuals, prepend=0) == -1)[0]

    # print(one_start_idx)

    assert len(one_start_idx) == len(zero_start_idx) + 1 or len(one_start_idx) == len(zero_start_idx)

    if len(one_start_idx) == len(zero_start_idx) + 1:
        zero_start_idx = np.append(zero_start_idx, len(predicts))
    # print(len(one_start_idx))

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
    #print(f'AREA UNDER CURVE {area}')
    return area_under_f1, max_f1_k, k_max, preds_for_max, fprs, tprs


def evaluate(score, label, validation_thresh=None):
    '''
    score:  (aka anomaly score) val_size
    labels: val_size
    return: ROC AUC,
            other metrics according to validation_thresh
                if validation_thresh, else: best metrics
    '''
    if len(score) < len(label):
        label = label[:len(label) - (len(label) - len(score))]
    elif len(score) > len(label):
        score = score[:len(score) - (len(score) - len(label))]
    
    false_pos_rates = []
    true_pos_rates = []
    f1s = []
    max_f1s_k = []
    preds = []

    assert score is not np.nan
    # thresholds = np.arange(0, score.max(), min(0.001, score.max()/50))#0.001
    # print(score.max())

    thresholds = np.arange(0, score.max(), score.max()/50)

    max_ks = []
    pairs = []


    for thresh in tqdm(thresholds):
        f1, max_f1_k, k_max, best_preds, fprs, tprs = pak_protocol(score, label, thresh)
        max_f1s_k.append(max_f1_k)
        max_ks.append(k_max)
        preds.append(best_preds)
        false_pos_rates.append(fprs)
        true_pos_rates.append(tprs)
        f1s.append(f1)
        pairs.extend([(thresh, i) for i in range(101)])
    
    if validation_thresh:
        f1, max_f1_k, max_k, best_preds, _, _ = pak_protocol(score, label, validation_thresh)
    else:    
        f1 = max(f1s)
        max_possible_f1 = max(max_f1s_k)
        max_idx = max_f1s_k.index(max_possible_f1)
        max_k = max_ks[max_idx]
        thresh_max_f1 = thresholds[max_idx]
        best_preds = preds[max_idx]
        best_thresh = thresholds[f1s.index(f1)]
    
    roc_max = metrics.auc(np.transpose(false_pos_rates)[max_k], np.transpose(true_pos_rates)[max_k])
    #np.save('/root/Diff-Anomaly/DiffusionAE/plots_for_paper/fprs_diff_score_pa.npy', np.transpose(false_pos_rates)[0])
    #np.save('/root/Diff-Anomaly/DiffusionAE/plots_for_paper/tprs_diff_score_pa.npy', np.transpose(true_pos_rates)[0])
    
    false_pos_rates = np.array(false_pos_rates).flatten()
    true_pos_rates = np.array(true_pos_rates).flatten()

    sorted_indexes = np.argsort(false_pos_rates) 
    false_pos_rates = false_pos_rates[sorted_indexes]
    true_pos_rates = true_pos_rates[sorted_indexes]
    pairs = np.array(pairs)[sorted_indexes]
    roc_score = metrics.auc(false_pos_rates, true_pos_rates)

    #np.save('/root/Diff-Anomaly/DiffusionAE/plots_for_paper/tprs_diff_score.npy', true_pos_rates)
    #np.save('/root/Diff-Anomaly/DiffusionAE/plots_for_paper/fprs_diff_score.npy', false_pos_rates)
    #np.save('/root/Diff-Anomaly/DiffusionAE/plots_for_paper/pairs_diff_score.npy', pairs)
    #preds = predictions[f1s.index(f1)]
    if validation_thresh:
        return {
            'f1': f1,   # f1_k(area under f1) for validation threshold
            'ROC/AUC': roc_score, # for all ks and all thresholds obtained on test scores
            'f1_max': max_f1_k, # best f1 across k values
            'preds': best_preds, # corresponding to best k 
            'k': max_k, # the k value correlated with the best f1 across k=1,100
            'thresh_max': validation_thresh,
            'roc_max': roc_score,
        }
    else:
        return {
            'f1': f1,  # f1_k(area under f1)
            'ROC/AUC': roc_score,
            'threshold': best_thresh,
            'f1_max': max_possible_f1, 
            'roc_max': roc_max,
            'thresh_max': thresh_max_f1, 
            # 'preds': best_preds,
            'k': max_k,
        }  # , false_pos_rates, true_pos_rates