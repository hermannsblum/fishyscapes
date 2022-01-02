from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from .common import get_observer, experiment_context, clear_directory, load_data
import os
import sys
import logging
from zipfile import ZipFile
import tensorflow as tf
import bdlb
from .utils import ExperimentData
from fs.data.fsdata import FSData
from fs.data.utils import load_gdrive_file
from fs.data.augmentation import crop_multiple
ex = Experiment()
ex.capture_out_filter = apply_backspaces_and_linefeeds
ex.observers.append(get_observer())


@ex.command
def saved_model(testing_dataset, model_id, _run, _log, batching=False, validation=False):
    fsdata = FSData(**testing_dataset)

    
    # Hacks because tf.data is shit and we need to translate the dict keys
    def data_generator():
        dataset = fsdata.validation_set if validation else fsdata.testset
        for item in dataset:
            data = fsdata._get_data(training_format=False, **item)
            out = {}
            for m in fsdata.modalities:
                blob = crop_multiple(data[m])
                if m == 'rgb':
                    m = 'image_left'
                if 'mask' not in fsdata.modalities and m == 'labels':
                    m = 'mask'
                out[m] = blob
            yield out

    
    data_types = {}
    for key, item in fsdata.get_data_description()[0].items():
        if key == 'rgb':
            key = 'image_left'
        if 'mask' not in fsdata.modalities and key == 'labels':
            key = 'mask'
        data_types[key] = item

    
    data = tf.data.Dataset.from_generator(data_generator, data_types)

    
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

    
    fs = bdlb.load(benchmark="fishyscapes", download_and_prepare=False)
    _run.info['{}_anomaly'.format(model_id)] = fs.evaluate(eval_func, data)


@ex.command
def resynthesis_model(testing_dataset, _run, _log, ours=True, validation=False):
    # added import inside the function to prevent conflicts if this method is not being tested
    sys.path.insert(0, os.path.join(os.getcwd(), os.path.dirname(os.path.dirname(__file__))))
    from test_fishy_torch import AnomalyDetector
    detector = AnomalyDetector(ours=ours)

    fsdata = FSData(**testing_dataset)

    # Hacks because tf.data is shit and we need to translate the dict keys
    def data_generator():
        dataset = fsdata.validation_set if validation else fsdata.testset
        for item in dataset:
            data = fsdata._get_data(training_format=False, **item)
            out = {}
            for m in fsdata.modalities:
                blob = crop_multiple(data[m])
                if m == 'rgb':
                    m = 'image_left'
                if 'mask' not in fsdata.modalities and m == 'labels':
                    m = 'mask'
                out[m] = blob
            yield out

    data_types = {}
    for key, item in fsdata.get_data_description()[0].items():
        if key == 'rgb':
            key = 'image_left'
        if 'mask' not in fsdata.modalities and key == 'labels':
            key = 'mask'
        data_types[key] = item

    data = tf.data.Dataset.from_generator(data_generator, data_types)

    fs = bdlb.load(benchmark="fishyscapes", download_and_prepare=False)

    if ours:
        model_id = 'SynBoost'
    else:
        model_id = 'Resynthesis'

    def wrapper(image):
        image = image.numpy().astype('uint8')
        ret = detector.estimator_worker(image)
        return ret

    _run.info['{}_anomaly'.format(model_id)] = fs.evaluate(wrapper, data)


if __name__ == '__main__':
    ex.run_commandline()
    os._exit(os.EX_OK)
