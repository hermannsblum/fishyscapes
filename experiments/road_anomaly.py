from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from .common import get_observer, experiment_context, clear_directory, load_data
import os
import sys
import logging
from zipfile import ZipFile
import tensorflow as tf
import bdlb
from bdlb.fishyscapes.benchmark_road import FishyscapesOnRoad_RODataset

from fs.data.utils import load_gdrive_file
from .utils import ExperimentData
from fs.data.augmentation import crop_multiple

ex = Experiment()
ex.capture_out_filter = apply_backspaces_and_linefeeds
ex.observers.append(get_observer())

@ex.command
def saved_model(testing_dataset, model_id, _run, _log, batching=False, validation=False):
    bench = FishyscapesOnRoad_RODataset()
    data = bench.get_dataset()

    ZipFile(load_gdrive_file(model_id, 'zip')).extractall('/tmp/extracted_module')
    tf.compat.v1.enable_resource_variables()
    net = tf.saved_model.load('/tmp/extracted_module')

    def eval_func(image):
        if batching:
            image = tf.expand_dims(image, 0)
        out = net.signatures['serving_default'](tf.cast(image, tf.float32))
        for key, val in out.items():
            print(key, val.shape, flush=True)
        return out['anomaly_score']

    _run.info['{}_anomaly'.format(model_id)] = bench.evaluate(eval_func, data)


if __name__ == '__main__':
    ex.run_commandline()
    os._exit(os.EX_OK)
