---
title: The Fishyscapes Benchmark
subtitle: Anomaly Detection for Semantic Segmentation
layout: page
callouts:
  - image_callout
  - subscribe_callout
show_sidebar: false
hide_footer: false
---

# Safe Deployment of Deep Learning on Robots

<div class="columns">
<div class="column has-text-centered">
<img width="400px" src="{{ "/assets/img/illustration.svg" | relative_url }}" />
</div>
<div class="column" markdown="1">
Research has produced Deep Learning methods that are increasingly accurate and start to generalise over illumination changes etc. However, modern networks are also known to be overconfident when exposed to anomalous or novel inputs.

The figure shows a prediction of DeepLabv3+, one of the leading methods for semantic segmentation in benchmarks like cityscapes, mapillary or the Robust Driving Benchmark. While the sheep does not fit into the set of classes it has been trained on, it very confidently assigns the classes street, human or sidewalk.

The Fishyscapes Benchmark compares research approaches towards detecting anomalies in the input. It therefore bridges another gap towards deploying learning systems on autonomous systems, that by definition have to deal with unexpected inputs and anomalies.
</div>
</div>
# Anomaly Detection for Semantic Segmentation
Anomaly detection and uncertainty estimation from deep learning models is subject of active research. However, fundamental ML research is often evaluated on MNIST, CIFAR and similar datasets. In the Fishyscapes Benchmark, we test how such methods transfer onto the much more complex task of semantic segmentation.
