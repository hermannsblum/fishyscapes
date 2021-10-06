---
title: Submission
# subtitle: This is the demo site for Bulma Clean Theme
layout: page
show_sidebar: false
hide_footer: false
---
To test methods on the dynamic datasets of the benchmark, we require submission of executables. These may be saved models, binaries, docker images or source code. Any submitted method should take an rgb image as input and produce semantic segmentation aswell as pixelwise uncertainty scores. See below for an example of submitting a tensorflow saved model and instructions for pytorch models.

<a class="button is-primary" target="_blank" href="https://forms.gle/Rrc9nWuxsmks9CuX7">Submit your Model</a>

# Submit a Tensorflow Graph
Save your model following the [tensorflow guide](https://www.tensorflow.org/guide/saved_model) with the following specifications of its input and output:

```python
tf.saved_model.simple_save(
    <session object>,
    <export path>,
    inputs={'rgb': <your rgb input tensor>},
    outputs={'prediction': <semantic classification of the input image>,
             'anomaly_score': <anomaly score tensor>})
```
The anomaly score tensor has the same height and width as the input image and assigns for each pixel a `float32` score that is higher for higher probability of anomaly. You do not have to threshold anything as we will test the methods over all possible thresholds.

<article class="message is-warning">
  <div class="message-header">
    <p>Important</p>
  </div>
  <div class="message-body">
    The saved model does not contain any original code of yours, only the tensorflow compute graph defined by the code.
  </div>
</article>

# Submit a Pytorch Model
Save and submit any model code that should be executed through the submission form above. Additionally, to correctly execute your model, submit a pull request to our [evaluation code](https://github.com/hermannsblum/fishyscapes/blob/master/experiments/fishyscapes.py). You should add a new function to the code following the examples that are already there and this skeleton code:

```python
@ex.command
def <your method name>(testing_dataset, _run, _log, validation=False):
    # BELOW, IMPORT ANY OF YOUR NETWORK CODE
    from <your package> import <your function>

    # SET UP YOUR NETWORK HERE

    fsdata = FSData(**testing_dataset)

    # BELOW CODE IS NECESSARY FROM US
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

    # FILL IN THE WRAPPER FUNCTION TO DO A SINGLE-FRAME PREDICTION
    def wrapper(image):
        image = image.numpy().astype('uint8')
        return my_anomaly_score(image)

    _run.info['<your method name>'] = fs.evaluate(wrapper, data)
```

Similar functions should also be added to the runners for `cityscapes_mIoU.py` and `timing.py`
