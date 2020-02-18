from os import path, environ, listdir
from zipfile import ZipFile
import json
import tensorflow as tf
import numpy as np
import cv2

from fs.settings import DATA_BASEPATH
from .data_baseclass import DataBaseclass


class FSData(DataBaseclass):

    def __init__(self, base_path, in_memory=False, num_classes=None, modalities=None,
                 **config):
        self.modalities = modalities
        if not base_path.startswith('/'):
            base_path = path.join(DATA_BASEPATH, base_path)
        if not path.exists(base_path):
            message = 'ERROR: Path to CITYSCAPES dataset does not exist.'
            print(message)
            raise IOError(1, message, base_path)

        self.base_path = base_path
        if in_memory:
            self.cache = {}

        def get_filenames(setname):
            if 'TMPDIR' in environ:
                print('INFO Loading %s into machine ... ' % setname, end='')
                with ZipFile(path.join(base_path, '%s.zip' % setname), 'r') as arch:
                    arch.extractall(path=environ['TMPDIR'])
                self.base_path = environ['TMPDIR']
                print('DONE')

            all_files = listdir(path.join(self.base_path, setname))
            if 'dataset_info.json' not in all_files:
                message = 'ERROR: %s does not contain dataset_info.json' % setname
                print(message)
                raise IOError(1, message, base_path)
            # load the info file and check whether it is consistent with info from other
            # sets
            with open(path.join(self.base_path, setname, 'dataset_info.json'), 'r') as f:
                info = json.load(f)
            if hasattr(self, 'datainfo'):
                if not self.datainfo == info:
                    raise UserWarning('ERROR: mismatching datainfo for {}: {} does not '
                                      'match previously loaded info {}'.format(
                                          setname, json.dumps(info),
                                          json.dumps(self.datainfo)))
            else:
                self.datainfo = info

            all_files.remove('dataset_info.json')
            # now group filenames by their prefixes
            grouped_by_idx = {}
            for filename in all_files:
                prefix = filename.split('_')[0]
                grouped_by_idx.setdefault(prefix, []).append(
                    path.join(self.base_path, setname, filename))
            return [{'filepaths': grouped_by_idx[idx]}
                    for idx in sorted(grouped_by_idx.keys())]

        filesets = {}
        for setname in ['trainset', 'measureset', 'validation_set', 'testset']:
            filesets[setname] = []
            if setname in listdir(base_path) or '%s.zip' % setname in listdir(base_path):
                filesets[setname] = get_filenames(setname)

        if num_classes is not None:
            self.datainfo['num_classes'] = num_classes
        else:
            num_classes = self.datainfo['num_classes']
        self._data_shape_description = self.datainfo['output_shapes']

        DataBaseclass.__init__(self, filesets['trainset'], filesets['measureset'],
                               filesets['testset'], self.datainfo.get('labelinfo', {}),
                               validation_set=filesets['validation_set'],
                               num_classes=self.datainfo.get('num_classes'))

    def get_data_description(self, num_classes=None, first_class_is_void=False):
        name_to_tf_type = {'int32': tf.int32, 'float32': tf.float32, 'string': tf.string}
        data_type_description = {m: name_to_tf_type[name]
                                 for m, name in self.datainfo['output_types'].items()}
        if num_classes is None:
            if 'num_classes' not in self.datainfo:
                raise UserWarning('ERROR: Need to specify number of classes.')
            num_classes = self.datainfo['num_classes']
        return (data_type_description, self.datainfo['output_shapes'], num_classes,
                first_class_is_void)

    def _get_data(self, filepaths, training_format=None):
        blob = {}
        for filepath in filepaths:
            components = filepath.split('_')
            modality, filetype = components[-1].split('.')
            if self.modalities is not None and modality not in self.modalities:
                # skip this modality
                continue
            if hasattr(self, 'cache'):
                if filepath in self.cache:
                    blob[modality] = self.cache[filepath].copy()
                    continue
            if filetype == 'npz':
                blob[modality] = np.load(filepath)[modality]
            elif filetype == 'npy':
                blob[modality] = np.load(filepath)
            elif modality == 'rgb':
                img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
                if img.shape[2] == 4:
                    rgb = img[:, :, :3][..., ::-1]
                    alpha = img[:, :, 3]
                    blob['rgb'] = np.concatenate(
                        (rgb, np.expand_dims(alpha, -1)), axis=-1)
                else:
                    blob['rgb'] = img[..., ::-1]
            else:
                data = cv2.imread(filepath, cv2.IMREAD_ANYDEPTH)
                if modality in ['labels', 'mask']:
                    # opencv translation as it ignores negative values
                    data = data.astype(self.datainfo['output_types'][modality])
                    data[data == 255] = -1
                blob[modality] = data
            if hasattr(self, 'cache'):
                self.cache[filepath] = blob[modality].copy()
        return blob
