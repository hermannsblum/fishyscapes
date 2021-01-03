from sacred import Experiment
from .common import experiment_context, clear_directory, load_data
import os
import sys
import logging
from zipfile import ZipFile
import tensorflow as tf
import bdlb
import cv2

from .utils import ExperimentData
from fs.data.fsdata import FSData
from fs.data.utils import load_gdrive_file
from fs.data.augmentation import crop_multiple

ex = Experiment()


@ex.command
def saved_model(image_path,
                name,
                model_id,
                testing_dataset=None,
                batching=None,
                validation=None):
    image = cv2.imread(image_path)

    ZipFile(load_gdrive_file(model_id,
                             'zip')).extractall('/tmp/extracted_module')
    tf.compat.v1.enable_resource_variables()
    net = tf.saved_model.load('/tmp/extracted_module')
    out = net.signatures['serving_default'](tf.cast(image, tf.float32)).numpy()

    min_val, max_val = out.min(), out.max()
    disp = (out - min_val) / (max_val - min_val)
    disp = 255 - (np.clip(disp, 0, 1) * 255).astype('uint8')

    directory, filename = os.path.split(image_path)
    cv2.imwrite(
        os.path.join(directory, f'{filename.split(".")[0]}_{name}.jpg'), disp)


if __name__ == '__main__':
    ex.run_commandline()
    os._exit(os.EX_OK)
