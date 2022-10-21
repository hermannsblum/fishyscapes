from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from experiments.common import get_observer, experiment_context, clear_directory, load_data
import os
import sys
import logging
from zipfile import ZipFile
import tensorflow as tf
import bdlb
from tqdm import tqdm
import tensorflow_datasets as tfds

from experiments.utils import ExperimentData
from fs.data.utils import load_gdrive_file
from fs.data.augmentation import crop_multiple

ex = Experiment()
ex.capture_out_filter = apply_backspaces_and_linefeeds
ex.observers.append(get_observer())

@ex.command
def saved_model(testing_dataset, model_id, _run, _log, batching=False):
    # testing_dataset is not used, but useful to keep config file consistent with other
    # tests
    data = tfds.load(name='cityscapes', split='validation',
                     data_dir='/cluster/work/riner/users/blumh/tensorflow_datasets')
    label_lookup = tf.constant(
        [-1, -1, -1, -1, -1, -1, -1, 0, 1, -1, -1, 2, 3, 4, -1, -1, -1, 5, -1, 6, 7, 8, 9,
         10, 11, 12, 13, 14, 15, -1, -1, 16, 17, 18])
    def label_lookup_map(batch):
        batch['segmentation_label'] = tf.gather_nd(
            label_lookup,
            tf.cast(batch['segmentation_label'], tf.int32))
        return batch
    data = data.map(label_lookup_map)
    if batching:
        data = data.batch(1)

    ZipFile(load_gdrive_file(model_id, 'zip')).extractall('/tmp/extracted_module')
    tf.compat.v1.enable_resource_variables()
    net = tf.saved_model.load('/tmp/extracted_module')

    m = tf.keras.metrics.MeanIoU(num_classes=19)
    for batch in tqdm(data, ascii=True):
        pred = net.signatures['serving_default'](tf.cast(batch['image_left'], tf.float32))
        labels = tf.reshape(batch['segmentation_label'], [-1])
        weights = tf.where(labels == -1, 0, 1)
        labels = tf.where(labels == -1, 0, labels)
        m.update_state(labels,
                       tf.reshape(pred['prediction'], [-1]),
                       sample_weight=weights)

    _run.info['mIoU'] = m.result().numpy()


@ex.command
def saved_model(testing_dataset, model_id, _run, _log, batching=False):
    ############## START ##############
    import os, sys, mmcv, torch
    from mmcv.cnn.utils.sync_bn import revert_sync_batchnorm
    MMSEG_DIR = os.environ.get('MMSEG_DIR')  # pointer to the provided "fmd" folder
    sys.path.append(os.path.abspath(MMSEG_DIR))
    print('Adding {} to sys path'.format(os.path.abspath(MMSEG_DIR)))
    from mmseg.apis import inference_segmentor, init_segmentor
    modelid = 'FMDC'
    config = 'local_configs/segformer/segformer.b2.1024x1024.cityf.{}.py'.format(modelid)
    if modelid in ['MSP', 'MLG', 'ENE', 'MCD', 'SML']:
        checkpoint = 'checkpoints/segformer_b2_BASE.pth'
    elif modelid in ['GMMU', 'GMMC', 'FMDU', 'FMDC', 'FMDU4L', 'FMDC4L']:
        checkpoint = 'checkpoints/segformer_b2_{}_tanh.pth'.format(modelid)
    else:
        raise NotImplementedError('{} is not supported model ID!'.format(modelid))

    gpu_id='0'
    map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(map_location + ':' + gpu_id if torch.cuda.is_available() else map_location)

    checkpoint_file = os.path.join(MMSEG_DIR, checkpoint)
    config_file = os.path.join(MMSEG_DIR, config)

    model = init_segmentor(config_file, checkpoint_file, device)
    model = revert_sync_batchnorm(model)
    ############## END ##############

    # testing_dataset is not used, but useful to keep config file consistent with other
    # tests
    data = tfds.load(name='cityscapes', split='validation',
                     data_dir='/cluster/work/riner/users/blumh/tensorflow_datasets')
    label_lookup = tf.constant(
        [-1, -1, -1, -1, -1, -1, -1, 0, 1, -1, -1, 2, 3, 4, -1, -1, -1, 5, -1, 6, 7, 8, 9,
         10, 11, 12, 13, 14, 15, -1, -1, 16, 17, 18])
    def label_lookup_map(batch):
        batch['segmentation_label'] = tf.gather_nd(
            label_lookup,
            tf.cast(batch['segmentation_label'], tf.int32))
        return batch
    data = data.map(label_lookup_map)
    if batching:
        data = data.batch(1)

    ZipFile(load_gdrive_file(model_id, 'zip')).extractall('/tmp/extracted_module')
    tf.compat.v1.enable_resource_variables()

    def eval_func(image):
        img = mmcv.imread(image)
        results = inference_segmentor(model, img)
        return results[0][0]

    m = tf.keras.metrics.MeanIoU(num_classes=19)
    for batch in tqdm(data, ascii=True):
        pred = eval_func(batch['image_left'])
        labels = tf.reshape(batch['segmentation_label'], [-1])
        weights = tf.where(labels == -1, 0, 1)
        labels = tf.where(labels == -1, 0, labels)
        m.update_state(labels,
                       tf.reshape(pred['prediction'], [-1]),
                       sample_weight=weights)

    _run.info['mIoU'] = m.result().numpy()


if __name__ == '__main__':
    ex.run_commandline()
    os._exit(os.EX_OK)
