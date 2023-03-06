---
title: Submission
# subtitle: This is the demo site for Bulma Clean Theme
layout: page
show_sidebar: false
hide_footer: false
---

<!--article class="message is-warning">
  <div class="message-header">
    <p>New Submission System</p>
  </div>
  <div class="message-body">
    We are working on a new submission system that will go online in early February 2023. To concentrate on getting it online soon, we currently do not accept new submissions in the old system.
    To prepare a submission, expect to upload a <a href="https://docs.sylabs.io/guides/3.10/user-guide/build_a_container.html">singularity image</a> that will mount a folder `/input` and should process all images to produce both segmentation and anomaly scores (as npy files) in `/output`.
  </div>
</article-->

# overview
To submit to fishyscapes, prepare a [apptainer container](https://apptainer.org/docs/user/latest/build_a_container.html) that will run your method on a mounted input folder. Once the container is started, it should process al images at `/input` and produce both segmentation and anomaly scores as `.npy` files in `/output`.

## container input and output requirements
The folder `/input` contains a number of files of the format `DDDD_*.png` where each `D` is a digit, e.g. `0000_04_Maurener_Weg_8_000000_000030_rgb.png0000_04_Maurener_Weg_8_000000_000030_rgb.png`. It is the same naming convention that is used in the validation set. The container should then save the output for each input in separate files: `/output/DDDD_anomaly.npy` should be a saved numpy array of the same resolution as the input image with per-pixel anomaly scores. `/output/DDDD_segmentation.npy` should be a saved numpy integer array of the same resolution as the input image with a per-pixel assigned class between 0 and 19.

## recommendations for working with singularity containers
We recommend to use docker as much as possible and only convert to singularity format as one of the last steps. A good starting point are e.g. the nvidia docker images or existing containers with pre-installed pytorch or tensorflow in the version that you require. You can find an example of how to create a submission container [here](https://github.com/4PiR2/fishyscapes_simg_example).

## submitting your container
Once you have a submittable `.simg` singularity container, please follow these steps:

1. Create a pull-request [here](https://github.com/hermannsblum/fishyscapes) where you edit the file `validation_performance.json` with your expected performance on the validation set. This will be used to validate that your uploaded container produces the same performance on our cluster.
2. Upload your container with [this form](https://docs.google.com/forms/d/e/1FAIpQLSca9dvNewWT-76BiKqm7ougH76Cb9SaB9nbkG5ckLOxQQIK4Q/viewform) and enter the number of your pull-request.
3. Re-run the validation job in github and check if the validation of the pull-request succeeds. If there are errors, you can find them in the github action log. Once you have fixed them, submit a new container following step 2. Repeat until the validation succeeds and you fixed all errors.
4. We will run your submitted model on the test sets and report the results on the website. You will receive a comment to your pull-request once the results are online.

## frequent errors

- Please make sure that your container image is below 10GB in total size. Google does not allow larger uploads yet.
- Your container should be compatible with apptainer version 1.1.5
