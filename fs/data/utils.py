import tensorflow as tf
import numpy as np
from os import path
import cv2
import json
from copy import copy


def array_from_data(dataset):
    """Creates a numpy array from a tf dataset."""
    results = []
    if tf.compat.v1.executing_eagerly():
        for batch in dataset:
            results.append({k: batch[k].numpy() for k in batch})
    else:
        with tf.Session() as sess:
            iterator = dataset.make_one_shot_iterator()
            next_element = iterator.get_next()
            try:
                while True:
                    results.append(sess.run(next_element))
            except tf.errors.OutOfRangeError:
                pass

    # merge list of dicts
    def merge(*values):
        if np.ndim(values[0]) == 0:
            return np.array(values)
        return np.stack(values, axis=0)

    return {k: merge(*(result[k] for result in results)) for k in results[0]}


def dump_dataset(dataset, directory: str, only_modalities=None, num_classes=None,
                 rgb_as_np=False, use_name=False):
    """Writes dataset to file structure."""
    if not path.exists(directory):
        message = 'ERROR: Path for dataset dump does not exist.'
        print(message)
        raise IOError(1, message, directory)

    def dump_blob(idx, blob):
        for m in blob:
            if only_modalities is not None and m not in only_modalities:
                continue
            if 'name' in blob:
                name = copy(blob['name'])
                name = name.decode() if hasattr(name, 'decode') else name
                filename = '{:04d}_{}_{}'.format(idx, name, m)
            else:
                filename = '{:04d}_{}'.format(idx, m)

            if m in ['rgb', 'visual', 'orig'] and not rgb_as_np:
                if blob[m].shape[2] == 4:
                    # convert to BGRA
                    bgr = blob[m].astype('uint8')[..., :3][..., ::-1]
                    a = blob[m].astype('uint8')[..., 3:]
                    data = np.concatenate((bgr, a), axis=-1)
                else:
                    # convert to BGR
                    data = blob[m].astype('uint8')[..., ::-1]
                cv2.imwrite(path.join(directory, filename + '.png'), data)
            elif (blob[m].ndim > 2 and blob[m].shape[2] > 1) \
                    or (m == 'rgb' and rgb_as_np):
                # save as numpy array
                np.savez_compressed(path.join(directory, filename + '.npz'),
                                    **{m: blob[m]})
            else:
                data = blob[m]
                # translate -1 values into something open-cv does not ignore
                data[data == -1] = 255
                data = data.astype('uint8')
                cv2.imwrite(path.join(directory, filename + '.png'), data)

    if tf.compat.v1.executing_eagerly():
        for idx, blob in enumerate(dataset):
            dump_blob(idx, {k: blob[k].numpy() for k in blob})
    else:
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        with tf.Session() as sess:
            try:
                idx = 0
                while True:
                    blob = sess.run(next_element)
                    dump_blob(blob)
                    idx += 1
            except tf.errors.OutOfRangeError:
                pass

    # write a description of the data
    info = {
        'output_shapes': {m: [shape_item for shape_item in shape]
                          for m, shape in dataset.output_shapes.items()
                          if not (only_modalities is not None
                                  and m not in only_modalities)},
        'output_types': {m: dtype.name for m, dtype in dataset.output_types.items()
                         if not (only_modalities is not None
                                 and m not in only_modalities)}}
    if num_classes is not None:
        info['num_classes'] = num_classes
    with open(path.join(directory, 'dataset_info.json'), 'w') as f:
        json.dump(info, f)
