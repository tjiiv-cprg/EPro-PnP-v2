# EPro-PnP-6DoF v2

<img src="viz.gif" width="550" alt=""/>

The code is modified from the [official implementation of CDPN](https://github.com/LZGMatrix/CDPN_ICCV2019_ZhigangLi), and is used for benchmarking only. **We will not maintain this code except for bug fixes.**

## Environment

The code has been tested in the environment described as follows:

- Linux (tested on Ubuntu 16.04/18.04)
- Python 3.6
- [PyTorch](https://pytorch.org/get-started/previous-versions/) 1.5.0

An example script for installing the python dependencies under CUDA 10.2:

```bash
# Create conda environment
conda create -y -n epropnp_6dof python=3.6
conda activate epropnp_6dof
conda install -y pip

# Install pytorch
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.2 -c pytorch

# Install other dependencies
pip install opencv-python==4.5.1.48 pyro-ppl==1.4.0 PyYAML==5.4.1 matplotlib termcolor plyfile easydict scipy progress numba tensorboardx
```

## Data preparation

Please refer to [this link](https://github.com/LZGMatrix/CDPN_ICCV2019_ZhigangLi#prepare-the-dataset).

## Models

The pretrained models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/10lUCUQKGGL7IEUHJ4KwWfnCyqbjTapQb).

## Train

If you use the `epropnp_v2_cdpn_init` config, please download the checkpoint `cdpn_stage_1.pth` from [[Google Drive](https://drive.google.com/drive/folders/1Jem2XsdHxr3ETRsZYqyTUmo5F3TmJGfO?usp=sharing) | [Baidu Pan](https://pan.baidu.com/s/19QxntwH22O4g2oYJWMBsLg?pwd=afa8)], and move it to `EPro-PnP-6DoF_v2/checkpoints/cdpn_stage_1.pth`.

To start training, enter the directory `EPro-PnP-6DoF_v2/tools`, and run:

```bash
python main.py --cfg /PATH/TO/CONFIG  # configs are located in EPro-PnP-6DoF_v2/tools/exp_cfg
```

By default GPU 0 is used, you can set the environment variable `CUDA_VISIBLE_DEVICES` to change this behavior.

Checkpoints, logs and visualizations will be saved to `EPro-PnP-6DoF_v2/exp`. You can run TensorBoard to plot the logs:

```bash
tensorboard --logdir ../exp
```

## Test

To test and evaluate on the LineMOD test split, please edit the config file and

1. set the `load_model` option to the path of the checkpoint file,
2. change the `test` option from `False` to `True`.

After saving the test config, enter the directory `EPro-PnP-6DoF_v2/tools`, and run:

```bash
python main.py --cfg /PATH/TO/CONFIG
```

Logs and visualizations will be saved to `EPro-PnP-6DoF_v2/exp`.
