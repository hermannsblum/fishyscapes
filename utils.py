import os
import numpy as np
from typing import List
import subprocess
from sklearn.metrics import confusion_matrix


def run(cmd, cwd=None, env=None):
    print(f'>>> {cwd} $ {" ".join(cmd)}')
    p = subprocess.Popen(cmd, stdin=None, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         cwd=cwd, env=env, universal_newlines=True, bufsize=1)
    os.set_blocking(p.stdout.fileno(), False)
    os.set_blocking(p.stderr.fileno(), False)
    lines = []
    lines_out = []
    lines_err = []
    while True:
        rc = p.poll()
        while line_out := p.stdout.readline():
            lines.append(line_out)
            lines_out.append(line_out)
            print(line_out, end='')
        while line_err := p.stderr.readline():
            lines.append(line_err)
            lines_err.append(line_err)
            print(line_err, end='')
        if rc is not None:
            break
    print('EXIT CODE:', rc, flush=True)
    assert rc == 0
    return rc, ''.join(lines), ''.join(lines_out), ''.join(lines_err)

def list_img_from_dir(data_dir: str, ext: str = '.png'):
    images = np.array([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith(f'{ext}')])
    order = np.argsort([int(p.split('/')[-1].split('_')[0]) for p in images])
    return images[order]

def calculate_metrics_perpixAP(labels: List[np.ndarray], uncertainties: List[np.ndarray], num_points=50):

    # concatenate lists for labels and uncertainties together
    if (labels[0].shape[-1] > 1 and np.ndim(labels[0]) > 2) or \
            (labels[0].shape[-1] == 1 and np.ndim(labels[0]) > 3):
        # data is already in batches
        labels = np.concatenate(labels)
        uncertainties = np.concatenate(uncertainties)
    else:
        labels = np.stack(labels)
        uncertainties = np.stack(uncertainties)
    labels = labels.squeeze()
    uncertainties = uncertainties.squeeze()

    # NOW CALCULATE METRICS
    pos = labels == 1
    valid = np.logical_or(labels == 1, labels == 0)  # filter out void
    gt = pos[valid]
    del pos
    uncertainty = uncertainties[valid].reshape(-1).astype(np.float32, copy=False)
    del valid

    # Sort the classifier scores (uncertainties)
    sorted_indices = np.argsort(uncertainty, kind='mergesort')[::-1]
    uncertainty, gt = uncertainty[sorted_indices], gt[sorted_indices]
    del sorted_indices

    # Remove duplicates along the curve
    distinct_value_indices = np.where(np.diff(uncertainty))[0]
    threshold_idxs = np.r_[distinct_value_indices, gt.size - 1]
    del distinct_value_indices, uncertainty

    # Accumulate TPs and FPs
    tps = np.cumsum(gt, dtype=np.uint64)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    del threshold_idxs

    # Compute Precision and Recall
    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = tps / tps[-1]
    # stop when full recall attained and reverse the outputs so recall is decreasing
    sl = slice(tps.searchsorted(tps[-1]), None, -1)
    precision = np.r_[precision[sl], 1]
    recall = np.r_[recall[sl], 0]
    average_precision = -np.sum(np.diff(recall) * precision[:-1])

    # select num_points values for a plotted curve
    interval = 1.0 / num_points
    curve_precision = [precision[-1]]
    curve_recall = [recall[-1]]
    idx = recall.size - 1
    for p in range(1, num_points):
        while recall[idx] < p * interval:
            idx -= 1
        curve_precision.append(precision[idx])
        curve_recall.append(recall[idx])
    curve_precision.append(precision[0])
    curve_recall.append(recall[0])
    del precision, recall

    if tps.size == 0 or fps[0] != 0 or tps[0] != 0:
        # Add an extra threshold position if necessary
        # to make sure that the curve starts at (0, 0)
        tps = np.r_[0., tps]
        fps = np.r_[0., fps]

    # Compute TPR and FPR
    tpr = tps / tps[-1]
    del tps
    fpr = fps / fps[-1]
    del fps

    # Compute AUROC
    auroc = np.trapz(tpr, fpr)

    # Compute FPR@95%TPR
    fpr_tpr95 = fpr[np.searchsorted(tpr, 0.95)]

    return {
        'auroc': auroc,
        'AP': average_precision,
        'FPR@95%TPR': fpr_tpr95,
        'recall': np.array(curve_recall),
        'precision': np.array(curve_precision),
    }


class MeanIoU:
    def __init__(self, num_labels, ignore_index):
        super().__init__()
        self.num_labels = num_labels
        self.ignore_index = ignore_index
        self.confmat = np.zeros((num_labels, num_labels), dtype=np.int64)

    def update(self, preds, labels):
        preds = np.asarray(preds).reshape([-1])
        labels = np.asarray(labels).reshape([-1])
        valid_points = labels != self.ignore_index
        preds = preds[valid_points]
        labels = labels[valid_points]
        self.confmat += confusion_matrix(labels, preds, labels=list(range(self.num_labels)))
        print(self.confmat, flush=True)

    def compute(self):
        intersection = np.diag(self.confmat)
        union = self.confmat.sum(0) + self.confmat.sum(1) - intersection
        iou = intersection.astype(float) / union.astype(float)
        iou[union == 0] = 0.0
        return (iou).mean()
