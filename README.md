# MGTANet

---

This is the official implementation of ***MGTANet***, a novel architecture for 3D object detection, which encodes LiDAR point cloud sequences acquired by multiple successive scans. The encoding process of the point cloud sequence is performed on two different time scales. We first design a short-term motion-aware voxel encoding that captures the short-term temporal changes of point clouds driven by the motion of objects in each voxel. We also propose long-term motion-guided bird's eye view (BEV) feature enhancement that adaptively aligns and aggregates the BEV feature maps obtained by the short-term voxel encoding by utilizing the dynamic motion context inferred from the sequence of the feature maps.

- **Title**: MGTANet: Encoding LiDAR Point Cloud Sequence Using Long Short-Term Motion-Guided Temporal Attention for 3D Object Detection [Paper]
- **Author**: Junho Koh*, Junhyung Lee*, Youngwoo Lee, Jaekyum Kim, Jun Won Choi (* indicates equal contribution)
- **Conference**: Accepted for the 37th AAAI Conference on Artificial Intelligence (AAAI), 2023.
- **More details**: [[**Homepage**](https://sites.google.com/view/junhokoh/aaai2023?authuser=0)] [[**Paper**](https://arxiv.org/abs/2212.00442)]

![overall_aaai2023_final.png](docs/img/overall_aaai2023_final.png)

## News

- [2022.10.18] The code for ***MGTANet*** has been merged into the codebase ***Focals Conv***
- [2022.11.19] The paper presenting ***MGTANet*** was accepted for ***AAAI2023***

## Experimental results

### nuScenes dataset

|  | mAP | NDS | download |
| --- | --- | --- | --- |
| CenterPoint | 58.82 | 66.35 | centerpoint_wgt.pth |
| CenterPoint + SA-VFE (short-term) | 59.86 | 67.04 | centerpoint_sa_vfe_wgt.pth |
| CenterPoint + MGTANet | 64.80 | 70.60 | mgtanet_wgt.pth |

Visualization for comparing of CenterPoint and MGTANet

![far_false-1.png](docs/img/far_false-1.png)

![far_heading-1.png](docs/img/far_heading-1.png)

## Getting started

### Common settings and notes

- The experiments are run with Pytorch 1.8.1, CUDA 11.1, and CuDNN 8.0
- The training is conducted on 4 Titan RTX 208

### Installation

1. Clone this repository

```bash
git clone https://github.com/HYjhkoh/MGTANet.git && cd MGTANet
```

1. Install the environment with Docker

We provide a Dockerfile to build an image

```bash
# build an image with Pytorch 1.8.1, CUDA 11.1
docker pull hyjhkoh/mgtanet:latest
```

Run it with

```bash
sudo nvidia-docker run -it --ipc=host -v {MGTANET_DIR}:/mnt/MGTANet -v {DATASET_DIR}:/mnt/dataset/nuscenes/v1.0-trainval --shm-size=512g hyjhkoh/mgtanet:latest /bin/bash 
```

Install the requirements

```bash
cd {MGTANET_PATH} && sh setup.sh
```

1. Prepare the datasets

Download and organize the official nuScenes.

Create data

```bash
cd {MGTANET_PATH}/tools
python tools/create_data.py nuscenes_data_prep --root_path=NUSCENES_TRAINVAL_DATASET_ROOT --version="v1.0-trainval" --nsweeps=10
```

*Note that we use sequential gt-sampling for training long-term model. Please download the [additional database file](https://drive.google.com/file/d/1h8d3hHUhnQCYpb02rIeqHvyWErrS3y44/view?usp=share_link)

*Note that we conduct sequence data for training long-term model. Please download [infos.pkl](https://drive.google.com/file/d/13bE8MgWV_r5iBAjWx9yx7rQooU-NBfn-/view?usp=share_link).

1. Download pre-trained models

If you want to directly evaluate the trained models we provide, please download them first.

### Evaluation

We provide the trained weight file, so you can just run with that. You can also use the model you trained.

```bash
# CONFIG = config file name
# WEIGHT_PATH = path for weight file
python -m torch.distributed.launch --nproc_per_node=1 ./tools/dist_test.py configs/nusc/mgtanet/$CONFIG.py --work_dir ./work_dirs/mgtanet/$CONFIG --checkpoint $WEIGHT_PATH
```

### Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py configs/nusc/voxelnet/$CONFIG.py
```

- Note that we use 4 GPUs (Titan RTX 2080) to train models.

## To do list

- [ ]  Config files and scripts for the test augmentation (double flip) in nuScenes test submission.
- [ ]  Results and models of MGTANet with online mode
- [ ]  Develop the MGTANet to Sensor fusion

## Citation

If you find this project useful in your research, please consider citing:

```bash
@article{koh2021joint,
  title={Joint 3D Object Detection and Tracking Using Spatio-Temporal Representation of Camera Image and LiDAR Point Clouds},
  author={Koh, Junho and Kim, Jaekyum and Yoo, Jinhyuk and Kim, Yecheol and Choi, Jun Won},
  journal={arXiv preprint arXiv:2112.07116},
  year={2021}
}
```

## Acknowledgement

- This work is built upon the [**CenterPoint**](https://github.com/tianweiy/CenterPoint). Please refer to the official github repositories, CenterPoint for more information.
- This README follows the style of **[Focals Conv](https://github.com/dvlab-research/FocalsConv)**.

## License

This project is released under the Apache 2.0 license.

## Related Repos

1. spconv
2. Deformable Conv
3. Deformable DETR
