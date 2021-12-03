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
def leek_anomaly(_run, _log):
    # added import inside the function to prevent conflicts if this method is not being tested
    sys.path.insert(0, os.path.join(os.getcwd(), os.path.dirname(os.path.dirname(__file__)), 'leek_anomaly'))
    
    from leek_anomaly import network_wrapper
    import torch
    from torchvision import transforms
    from PIL import Image
    import numpy as np

    def val_data_transforms(mean_train=[0.485, 0.456, 0.406], std_train=[0.229, 0.224, 0.225]):
        data_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean_train,
                                    std=std_train)
                ])
        return data_transforms
    val_data_transform = val_data_transforms()

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

    m = tf.keras.metrics.MeanIoU(num_classes=19)
    for batch in tqdm(data, ascii=True):

        image = batch['image_left'].numpy().astype('uint8')
        test_img = Image.fromarray(np.array(image))
        test_img = val_data_transform(test_img)
        test_img = torch.unsqueeze(test_img, 0).to(network_wrapper.device)
        network_wrapper.features_t = []
        with torch.set_grad_enabled(False):
            out_logits, anomaly_score = network_wrapper.net(test_img)
        _, pred = out_logits.detach().cpu().max(1)

        labels = tf.reshape(batch['segmentation_label'], [-1])
        weights = tf.where(labels == -1, 0, 1)
        labels = tf.where(labels == -1, 0, labels)
        m.update_state(labels,
                       tf.reshape(pred.clone(), [-1]),
                       sample_weight=weights)


    _run.info['mIoU'] = m.result().numpy()
    

if __name__ == '__main__':
    ex.run_commandline()
    os._exit(os.EX_OK)
