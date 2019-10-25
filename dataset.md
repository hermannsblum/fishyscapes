---
title: Dataset
layout: page
show_sidebar: false
hide_footer: false
---
While most of the datasets remain on the evaluation servers to test methods for truely unknown objects, the FS Lost & Found validation set is publicly available as part of the BDL-Benchmarks framework. BDL Benchmarks is an open-source framework that aims to bridge the gap between the design of deep probabilistic machine learning models and their application to real-world problems, hosted on [GitHub](https://github.com/OATML/bdl-benchmarks).

```
pip install git+https://github.com/hermannsblum/bdl-benchmark.git
```

The framework automatically downloads the data and makes it easy to test your method:

<a class="button is-primary" target="_blank" href="https://github.com/hermannsblum/bdl-benchmark/blob/master/notebooks/fishyscapes.ipynb">See Notebook</a> <a class="button is-warning" target="_blank" href="https://colab.research.google.com/github/hermannsblum/bdl-benchmark/blob/master/notebooks/fishyscapes.ipynb">Run in Colab</a>

```python
import bdlb

fs = bdlb.load(benchmark="fishyscapes")
# automatically downloads the dataset
data = fs.get_dataset()

# test your method with the benchmark metrics
def estimator(image):
    """Assigns a random uncertainty per pixel."""
    uncertainty = tf.random.uniform(image.shape[:-1])
    return uncertainty

metrics = fs.evaluate(estimator, data.take(2))
print('My method achieved {:.2f}% AP'.format(100 * metrics['AP']))
```
