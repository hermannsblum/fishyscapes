import json
import os
from typing import List

import numpy as np
from PIL import Image
from utils import calculate_metrics_perpixAP, list_img_from_dir


def main():
    with open('settings.json', 'r') as f:
        settings = json.load(f)
    with open('validation_performance.json', 'r') as f:
        settings.update(json.load(f))

    path_labels = list_img_from_dir(settings['val_labels_path'], '_labels.png')
    path_uncertainties = list_img_from_dir(settings['tmp_pred_path'], '.npy')
    im_labels = [np.asarray(Image.open(p)) for p in path_labels]
    im_uncertainties = [np.load(p) for p in path_uncertainties]

    ret = calculate_metrics_perpixAP(im_labels, im_uncertainties)
    print(ret)

    # threshold for numerical errors
    eps = 0.001
    assert ret['AP'] >= settings['ap'] - eps and ret['AP'] <= settings['ap'] + eps
    assert ret['FPR@95%TPR'] >= settings['fpr'] - eps and ret['FPR@95%TPR'] <= settings['fpr'] + eps
    print('Successfully Validated !!!')


if __name__ == '__main__':
    main()
