# Anonymous Submission 

## Install 

1. Create a conda environment

    ```bash
    conda create -n fsbm python=3.8
    ```

2. Install numpy 1.22.3 and torch 1.7.1 with the CUDA version of the specific machine. For example,

    ```bash
    pip install numpy==1.22.3
    conda install -y pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=11.0 -c pytorch
    ```

3.  Install package dependencies

    ```bash
    pip install -r requirements.txt
    ```
4. Hopefully, these steps would create an environment identical 

## About code

We added a function `` to `experiments/fishyscapes.py`。In the `get_score` function in line 292, we expect an `ndarray` of shape `(H, W, 3)`. The return value of this function is `torch.Tensor` in shape `(H, W)` on CPU.

We perform multi-scale inferece on images, which cause the code to run a bit slower. It typically takes about 7 minutes and approximately 15 GB GPU memory to calculate the anormaly score for 100 images in the validation set.

