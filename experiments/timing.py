from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from .common import get_observer, experiment_context, clear_directory, load_data
import os
import sys
import logging
from zipfile import ZipFile
import tensorflow as tf
import tensorflow_datasets as tfds
import bdlb
from tqdm import tqdm
import time

from .utils import ExperimentData
from fs.data.fsdata import FSData
from fs.data.utils import load_gdrive_file
from fs.data.augmentation import crop_multiple

ex = Experiment()
ex.capture_out_filter = apply_backspaces_and_linefeeds
ex.observers.append(get_observer())


@ex.command
def saved_model(testing_dataset, model_id, _run, _log, batching=False, validation=False):
    data = tfds.load(name='cityscapes', split='validation',
                     data_dir='/cluster/work/riner/users/blumh/tensorflow_datasets')
    if batching:
        data = data.batch(1)
    data = data.prefetch(500)

    ZipFile(load_gdrive_file(model_id, 'zip')).extractall('/tmp/extracted_module')
    tf.compat.v1.enable_resource_variables()
    net = tf.saved_model.load('/tmp/extracted_module')

    def eval_func(image):
        out = net.signatures['serving_default'](tf.cast(image, tf.float32))
        return out['anomaly_score'], out['prediction']

    m = tf.keras.metrics.Mean()
    for batch in tqdm(data, ascii=True):
        start = time.time()
        eval_func(batch['image_left'])
        end = time.time()
        m.update_state(end - start)

    _run.info['{}_anomaly'.format(model_id)] = m.result().numpy()


@ex.command
def resynthesis_model(_run, _log, ours=True, validation=False):
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    # added import inside the function to prevent conflicts if this method is not being tested
    sys.path.insert(0, os.path.join(os.getcwd(), os.path.dirname(os.path.dirname(__file__)), 'driving_uncertainty'))
    from driving_uncertainty.test_fishy_torch import AnomalyDetector
    detector = AnomalyDetector(ours=ours)

    data = tfds.load(name='cityscapes', split='validation',
                     data_dir='/cluster/work/riner/users/blumh/tensorflow_datasets')

    if ours:
        model_id = 'SynBoost'
    else:
        model_id = 'Resynthesis'

    def eval_func(image):
        return detector.estimator_worker(image)

    m = tf.keras.metrics.Mean()
    for batch in tqdm(data, ascii=True):
        image = batch['image_left'].numpy().astype('uint8')
        start = time.time()
        eval_func(image).numpy()
        end = time.time()
        m.update_state(end - start)

    _run.info['{}_anomaly'.format(model_id)] = m.result().numpy()


@ex.command
def leek_anomaly(_run, _log, validation=False):
    # added import inside the function to prevent conflicts if this method is not being tested
    sys.path.insert(0, os.path.join(os.getcwd(), os.path.dirname(os.path.dirname(__file__)), 'leek_anomaly'))
    from leek_anomaly import network_wrapper
    
    data = tfds.load(name='cityscapes', split='validation',
                     data_dir='/cluster/work/riner/users/blumh/tensorflow_datasets')

    def eval_func(image):
        return network_wrapper.estimator(image)

    m = tf.keras.metrics.Mean()
    for batch in tqdm(data, ascii=True):
        image = batch['image_left'].numpy().astype('uint8')
        start = time.time()
        eval_func(image).numpy()
        end = time.time()
        m.update_state(end - start)

    _run.info['leek_anomaly'] = m.result().numpy()
    
if __name__ == '__main__':
    ex.run_commandline()
    os._exit(os.EX_OK)
