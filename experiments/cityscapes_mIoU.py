from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from .common import get_observer, experiment_context, clear_directory, load_data
import os
import sys
import logging
from zipfile import ZipFile
import tensorflow as tf
import bdlb
from tqdm import tqdm
import tensorflow_datasets as tfds

from .utils import ExperimentData
from fs.data.utils import load_gdrive_file
from fs.data.augmentation import crop_multiple

ex = Experiment()
ex.capture_out_filter = apply_backspaces_and_linefeeds
ex.observers.append(get_observer())

@ex.command
def saved_model(testing_dataset, model_id, _run, _log):
    # testing_dataset is not used, but useful to keep config file consistent with other
    # tests
    data = tfds.load(name='cityscapes', split='validation',
                     data_dir='/cluster/work/riner/users/blumh/tensorflow_datasets')
    data = data.prefetch(100)

    ZipFile(load_gdrive_file(model_id, 'zip')).extractall('/tmp/extracted_module')
    tf.compat.v1.enable_resource_variables()
    net = tf.saved_model.load('/tmp/extracted_module')

    m = tf.keras.metrics.MeanIoU(num_classes=19)
    for batch in tqdm(data, ascii=True):
        pred = net.signatures['serving_default'](tf.cast(batch['image_left'], tf.float32))
        m.update_state(tf.reshape(batch['segmentation_label'], [-1]),
                       tf.reshape(pred['prediction'], [-1]))

    _run.info['mIoU'] = m.result().numpy()


if __name__ == '__main__':
    ex.run_commandline()
    os._exit(os.EX_OK)
