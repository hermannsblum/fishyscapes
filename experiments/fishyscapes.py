from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from .common import get_observer, experiment_context, clear_directory, load_data
import os
import sys
import logging
from zipfile import ZipFile

from .utils import ExperimentData
from fs.data.fsdata import FSData

ex = Experiment()
ex.capture_out_filter = apply_backspaces_and_linefeeds
ex.observers.append(get_observer())

@ex.command
def saved_model(testing_dataset, model_id, _run, _log):
    data = FSData(**testing_dataset).get_testset()

    def translate_data_format(blob):
        blob['image_left'] = blob.get('rgb', blob['image_left'])
        return blob

    data.map(translate_data_format)

    ZipFile(load_gdrive_file(model_id, 'zip')).extractall('/tmp/extracted_module')
    tf.compat.v1.enable_resource_variables()
    net = tf.saved_model.load('/tmp/extracted_module')

    def eval_func(image):
        out = net.signatures['serving_default'](tf.cast(image, tf.float32))
        return out['anomaly_score']

    fs = bdlb.load(benchmark="fishyscapes", download_and_prepare=False)
    _run.info['{}_anomaly'.format(model_id)] = fs.evaluate(eval_func, data)


if __name__ == '__main__':
    ex.run_commandline()
    os._exit(os.EX_OK)
