---
title: Dataset
layout: page
show_sidebar: false
hide_footer: false
---
While most of the datasets remain on the evaluation servers to test methods for truely unknown objects, the FS Lost & Found validation set is publicly available.

<a class="button is-primary" target="_blank" href="http://robotics.ethz.ch/~asl-datasets/Fishyscapes/fishyscapes_lostandfound.zip">download FS Lost & Found validation set</a>

Below we document code that integrates the dataset with TFDS and BDL-Benchmark. This will also allow to download a small validation set of FS Static. We can not provide a zip download for FS Static since we are not allowed to host the Cityscapes data. Our code automatically generates the dataset from Cityscapes.

```
pip install git+https://github.com/hermannsblum/bdl-benchmark.git
```

The framework automatically downloads the data and makes it easy to test your method:

<a class="button is-primary" target="_blank" href="https://github.com/hermannsblum/bdl-benchmark/blob/master/notebooks/fishyscapes.ipynb">Notebook Source</a> <a class="button is-warning" target="_blank" href="https://colab.research.google.com/github/hermannsblum/bdl-benchmark/blob/master/notebooks/fishyscapes.ipynb">Run in Colab</a>

```python
import bdlb

fs = bdlb.load(benchmark="fishyscapes")
# automatically downloads the dataset
data = fs.get_dataset('LostAndFound')

# test your method with the benchmark metrics
def estimator(image):
    """Assigns a random uncertainty per pixel."""
    uncertainty = tf.random.uniform(image.shape[:-1])
    return uncertainty

metrics = fs.evaluate(estimator, data.take(2))
print('My method achieved {:.2f}% AP'.format(100 * metrics['AP']))
```

# FS Web Validation Data

The FS Web Dataset is regularly changing to model an open world setting. We make validation data available that is generated with the same image blending mechanisms, but instead of using dynaimc data from the web it uses objects from PASCAL VOC (see [paper](https://arxiv.org/pdf/1904.03215.pdf) for details). The dataset is intended to illustrate blending changes as the dataset evolves and enable contributors to test their methods on a dataset that is closer to the FS Web data.

<a class="button is-primary" target="_blank" href="https://github.com/hermannsblum/bdl-benchmark/blob/master/notebooks/fishyscapes web validation data.ipynb">Notebook Source</a> <a class="button is-warning" target="_blank" href="https://colab.research.google.com/github/hermannsblum/bdl-benchmark/blob/master/notebooks/fishyscapes web validation data.ipynb">Run in Colab</a>

# Attribution

When using the Lost & Found dataset, please make sure you correctly attribute it to the original authors:
```
@inproceedings{pinggera2016lost,
  title={Lost and found: detecting small road hazards for self-driving vehicles},
  author={Pinggera, Peter and Ramos, Sebastian and Gehrig, Stefan and Franke, Uwe and Rother, Carsten and Mester, Rudolf},
  booktitle={2016 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2016}
}
```

When using the segmentation masks, please also attribute these to the fishyscapes benchmark:
```
@article{blum2019fishyscapes,
  title={The Fishyscapes Benchmark: Measuring Blind Spots in Semantic Segmentation},
  author={Blum, Hermann and Sarlin, Paul-Edouard and Nieto, Juan and Siegwart, Roland and Cadena, Cesar},
  journal={arXiv preprint arXiv:1904.03215},
  year={2019}
}
```
