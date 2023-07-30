# EPro-PnP-Det v2

<img src="resources/viz.gif" alt=""/>

## Installation

Please refer to [INSTALL.md](INSTALL.md).

## Data Preparation

To train and evaluate the model, download the [full nuScenes dataset (v1.0)](https://www.nuscenes.org/nuscenes#download). Only the keyframe subset and metadata are required.

Create the directory `EPro-PnP-Det/data`. Extract the downloaded archives and symlink the dataset root to `EPro-PnP-Det/data/nuscenes` according to the following structure. If your folder structure is different, you may need to change the corresponding paths in config files.

```
EPro-PnP-Det_v2/
├── configs/
├── data/
│   └── nuscenes/
│       ├── maps/
│       ├── samples/
│       ├── v1.0-test/
│       └── v1.0-trainval/
├── demo/
├── epropnp_det/
├── resources/
├── tools/
…
```

Run the following commands to pre-process the data:

```bash
python tools/data_converter/nuscenes_converter.py data/nuscenes --version v1.0-trainval
# optionally if you want to evaluate on the test set
python tools/data_converter/nuscenes_converter.py data/nuscenes --version v1.0-test
```

Note that our data converter is different from MMDetection3D and EPro-PnP-Det v1, although they seem alike. If you have already converted the data in MMDetection3D's format, you still have to do another conversion in our format, which will not conflict with MMDetection3D. If you have already converted the data in EPro-PnP-Det v1's format, this conversion will overwrite the previous results.

## Models

All models are trained for 12 epochs on the nuScenes dataset using 2 RTX 3090 GPUs. Checkpoints are available for download at [Google Drive](https://drive.google.com/drive/folders/1D24OSyMPCifzOeowlltRGHN4m1vhDrTx).

## Test

To test and evaluate on the validation split, run:

```bash
python test.py /PATH/TO/CONFIG /PATH/TO/CHECKPOINT --val-set --eval nds
```

You can specify the GPUs to use by adding the `--gpu-ids` argument, e.g.:

```bash
python test.py /PATH/TO/CONFIG /PATH/TO/CHECKPOINT --val-set --eval nds --gpu-ids 0 1 2 3  # distributed test on 4 GPUs
```

To enable test-time augmentation (TTA), edit the configuration file and replace the string `flip=False` with `flip=True`.

To test on the test split and save the detection results, run:

```bash
python test.py /PATH/TO/CONFIG /PATH/TO/CHECKPOINT --format-only --eval-options jsonfile_prefix=/PATH/TO/OUTPUT/DIRECTORY
```

You can append the argument `--show-dir /PATH/TO/OUTPUT/DIRECTORY` to save visualized results.

To view other testing options, run:

```bash
python test.py -h
```

## Train

Run:

```bash
python train.py /PATH/TO/CONFIG --gpu-ids 0 1
```

Note that the total batch size is determined by the number of GPUs you specified. For EPro-PnP-Det v2 we use 2 GPUs, each processing 6 images. We recommend GPUs with at least 24 GB of VRAM. You may edit the `samples_per_gpu` option in the config file to vary the number of images per GPU.

To view other training options, run:

```bash
python train.py -h
```

By default, logs and checkpoints will be saved to `EPro-PnP-Det/work_dirs`. You can run TensorBoard to plot the logs:

```bash
tensorboard --logdir work_dirs
```

## Inference Demo

We provide a demo script to perform inference on images in a directory and save the visualized results. Example:

```bash
python demo/infer_imgs.py /PATH/TO/DIRECTORY /PATH/TO/CONFIG /PATH/TO/CHECKPOINT --intrinsic demo/nus_cam_front.csv --show-views 3d bev mc
```

The resulting visualizations will be saved into `/PATH/TO/DIRECTORY/viz`.

Another useful script is for visualizing an entire sequence from the nuScenes dataset, so that you can create video clips from the frames. Run the following command for more information:

```bash
python demo/infer_nuscenes_sequence.py -h
```
