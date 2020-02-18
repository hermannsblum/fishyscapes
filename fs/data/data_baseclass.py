import numpy as np
from random import Random
from sklearn.model_selection import train_test_split
import tensorflow as tf

from .utils import array_from_data
from .augmentation import crop_multiple


class DataBaseclass():
    """A basic, abstract class for splitting data into batches, compliant with DataWrapper
    interface."""

    def __init__(self, trainset, measureset, testset, labelinfo,
                 validation_set=None, num_classes=None, info=False):
        if validation_set is None:
            self.trainset, self.validation_set = train_test_split(
                trainset, test_size=15, random_state=317243896)
        else:
            self.trainset = trainset
            self.validation_set = validation_set
        self.measureset = measureset
        self.testset = testset
        if num_classes is None:
            self.num_classes = self._num_default_classes
        else:
            self.num_classes = num_classes
        self.modalities = list(self._data_shape_description.keys())
        self.labelinfo = labelinfo
        self.print_info = info
        Random(3).shuffle(self.trainset)

    @classmethod
    def get_data_description(cls, num_classes=None):
        """Produces a descriptor of the data the given class produces.

        For implementation reasons, this has to be a class method and cannot have
        access to object configurations (weird tensorflow errors if the dataclass
        is initialized before the model).

        Returns a tuple of 4 properties:
        - dict of modalities and their data types
        - dict of modalities and their data shapes
        - number of classes
        - True if the first class is void and shoul be ignored in evaluation

        If you will later modify the number of classes, please specify a manipulated
        number of classes as an optional argument.
        """
        data_shape_description = cls._data_shape_description
        modalities = list(data_shape_description.keys())
        if num_classes is None:
            num_classes = cls._num_default_classes
        first_class_is_void = True
        if hasattr(cls, '_first_class_is_void'):
            first_class_is_void = cls._first_class_is_void
        if hasattr(cls, '_data_type_description'):
            data_type_description = cls._data_type_description
        else:
            data_type_description = {
                'labels': tf.int32,
                **{m: tf.float32 for m in modalities if not m == 'labels'}}
        return (data_type_description, data_shape_description, num_classes,
                first_class_is_void)

    def _get_data(self, **kwargs):
        """Returns data for one item in trainset or testset. kwargs is the unfolded dict
        from the trainset or testset list
        # (it is called as self._get_data(one_hot=something, **testset[some_idx]))
        """
        raise NotImplementedError

    def _get_set_data(self, datasplit, training_format=False, tf_dataset=True,
                      num_items=None):
        if num_items is None:
            num_items = len(datasplit)
        if num_items > len(datasplit):
            raise UserWarning('ERROR: Requested more items than available.')

        def data_generator():
            for item in datasplit[:num_items]:
                data = self._get_data(training_format=training_format, **item)
                for m in self.modalities:
                    data[m] = crop_multiple(data[m])
                yield data

        dataset = tf.data.Dataset.from_generator(data_generator,
                                                 *self.get_data_description()[:2])
        if not tf_dataset:
            return array_from_data(dataset)
        return dataset

    def get_trainset(self, tf_dataset=True, training_format=True):
        """Return trainingset. By default as tf.data.dataset, otherwise as numpy array.
        """
        return self._get_set_data(self.trainset, tf_dataset=tf_dataset,
                                  training_format=training_format)

    def get_testset(self, num_items=None, tf_dataset=True):
        """Return testset. By default as tf.data.dataset, otherwise as numpy array."""
        return self._get_set_data(self.testset, tf_dataset=tf_dataset,
                                  num_items=num_items)

    def get_measureset(self, tf_dataset=True):
        """Return measureset. By default as tf.data.dataset, otherwise as numpy array."""
        return self._get_set_data(self.measureset, tf_dataset=tf_dataset)

    def get_validation_set(self, num_items=None, tf_dataset=True):
        """Return testset. By default as tf.data.dataset, otherwise as numpy array."""
        return self._get_set_data(self.validation_set, tf_dataset=tf_dataset,
                                  num_items=num_items)

    def coloured_labels(self, labels):
        """Return a coloured picture according to set label colours."""
        # To efficiently map class label to color, we create a lookup table
        lookup = np.array([self.labelinfo[i]['color']
                           for i in self.labelinfo.keys()]).astype(int)
        return np.array(lookup[labels[:]]).astype('uint8')
