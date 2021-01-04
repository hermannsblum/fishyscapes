from sacred import Experiment
from .common import experiment_context, clear_directory, load_data
import os
import sys
import logging
from zipfile import ZipFile
import tensorflow as tf
import bdlb
import cv2
import numpy as np

from .utils import ExperimentData
from fs.data.fsdata import FSData
from fs.data.utils import load_gdrive_file
from fs.data.augmentation import crop_multiple

ex = Experiment()


@ex.command
def saved_model(image_path,
                name,
                model_id,
                scale='linear',
                labels=None,
                testing_dataset=None,
                batching=False,
                validation=None):
    image = cv2.imread(image_path)

    ZipFile(load_gdrive_file(model_id,
                             'zip')).extractall('/tmp/extracted_module')
    tf.compat.v1.enable_resource_variables()
    net = tf.saved_model.load('/tmp/extracted_module')
    def eval_func(image):
        if batching:
            image = tf.expand_dims(image, 0)
        out = net.signatures['serving_default'](tf.cast(image, tf.float32))
        for key, val in out.items():
            print(key, val.shape, flush=True)
        return out['anomaly_score']
    out = eval_func(image).numpy()
    if batching:
        out = out[0]

    min_val, max_val = out.min(), out.max()
    disp = (out - min_val) / (max_val - min_val)
    disp = 255 - (np.clip(disp, 0, 1) * 255).astype('uint8')

    directory, filename = os.path.split(image_path)

    if labels is None:
        cv2.imwrite(
            os.path.join(directory, f'{filename.split(".")[0]}_{name}.jpg'), disp)
        return
    # since we have labels, check for the accuracy
    def data_generator():
        rgb = cv2.imread(image_path).astype('float32')
        label = cv2.imread(labels, cv2.IMREAD_ANYDEPTH).astype('int32')
        yield {'image_left': rgb, 'mask': label}
    data = tf.data.Dataset.from_generator(
        data_generator, {'image_left': tf.float32, 'mask': tf.int32})
    fs = bdlb.load(benchmark='fishyscapes', download_and_prepare=False)
    metrics = fs.evaluate(eval_func, data)
    print(metrics['AP'], flush=True)
    cv2.imwrite(
        os.path.join(directory,
                     f'{filename.split(".")[0]}_{name}_AP{100 * metrics["AP"]:.2f}.jpg'), disp)


@ex.command
def resynthesis_model(image_path,
                      name,
                      ours,
                      labels=None,
                      testing_dataset=None,
                      batching=None,
                      validation=None):
    # added import inside the function to prevent conflicts if this method is not being tested
    sys.path.insert(0, os.path.join(os.getcwd(), os.path.dirname(os.path.dirname(__file__)), 'driving_uncertainty'))
    from driving_uncertainty.test_fishy_torch import AnomalyDetector
    detector = AnomalyDetector(ours=ours)

    image = cv2.imread(image_path).astype('uint8')
    out = detector.estimator_worker(image).numpy()

    min_val, max_val = out.min(), out.max()
    disp = (out - min_val) / (max_val - min_val)
    disp = 255 - (np.clip(disp, 0, 1) * 255).astype('uint8')

    directory, filename = os.path.split(image_path)
    if labels is None:
        cv2.imwrite(
            os.path.join(directory, f'{filename.split(".")[0]}_{name}.jpg'), disp)
        return
    # since we have labels, check for the accuracy
    def data_generator():
        rgb = cv2.imread(image_path).astype('float32')
        label = cv2.imread(labels, cv2.IMREAD_ANYDEPTH).astype('int32')
        yield {'image_left': rgb, 'mask': label}
    data = tf.data.Dataset.from_generator(
        data_generator, {'image_left': tf.float32, 'mask': tf.int32})
    def eval_func(image):
        image = image.numpy().astype('uint8')
        ret = detector.estimator_worker(image)
        return ret
    fs = bdlb.load(benchmark='fishyscapes', download_and_prepare=False)
    metrics = fs.evaluate(eval_func, data)
    print(metrics['AP'], flush=True)
    cv2.imwrite(
        os.path.join(directory,
                     f'{filename.split(".")[0]}_{name}_AP{100 * metrics["AP"]:.2f}.jpg'), disp)

@ex.command
def segmentation(image_path):
    image = cv2.imread(image_path)

    ZipFile(load_gdrive_file('12ONfO6WIS16xkfu6ucHEy4_5Tre0yxC5',
                             'zip')).extractall('/tmp/extracted_module')
    tf.compat.v1.enable_resource_variables()
    net = tf.saved_model.load('/tmp/extracted_module')

    # batch processing
    image = tf.expand_dims(image, 0)
    out = net.signatures['serving_default'](tf.cast(image, tf.float32))
    out = out['prediction'].numpy()[0]

    # map colors
    color_map=np.array([[128, 64, 128],
            [244, 35, 232],
            [70, 70, 70],
            [102, 102, 156],
            [190, 153, 153],
            [153, 153, 153],
            [250, 170, 30],
            [220, 220, 0],
            [107, 142, 35],
            [152, 251, 152],
            [70, 130, 180],
            [220, 20, 60],
            [255, 0, 0],
            [0, 0, 142],
            [0, 0, 70],
            [0, 60, 100],
            [0, 80, 100],
            [0, 0, 230],
            [119, 11, 32]]).astype('int')
    disp = color_map[out].astype('uint8')[..., ::-1]  # convert to BGR for cv2
    directory, filename = os.path.split(image_path)
    cv2.imwrite(
        os.path.join(directory, f'{filename.split(".")[0]}_pred.png'), disp)

if __name__ == '__main__':
    ex.run_commandline()
    os._exit(os.EX_OK)
