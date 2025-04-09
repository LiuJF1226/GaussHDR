<p align="center">
<h1 align="center"><strong>GaussHDR: High Dynamic Range Gaussian Splatting via Learning Unified 3D and 2D Local Tone Mapping</strong></h1>
<h3 align="center">CVPR 2025</h3>

<p align="center">
    <a href="https://scholar.google.com/citations?hl=en&user=-moPItwAAAAJ">Jinfeng Liu</a><sup>1</sup>,</span>
    <a href="https://scholar.google.com/citations?hl=en&user=KKzKc_8AAAAJ">Lingtong Kong</a><sup>2</sup>,
    <a href="https://libraboli.github.io/">Li Bo</a><sup>2</sup>,
    <a href="https://www.danxurgb.net/">Dan Xu</a><sup>1</sup>
    <br>
        <sup>1</sup>HKUST,
        <sup>2</sup>vivo Mobile Communication Co., Ltd
</p>

<div align="center">
    <a href='https://arxiv.org/abs/2503.10143'><img src='https://img.shields.io/badge/ArXiv-Paper-b31b1b.svg'></a>  
    <a href='https://liujf1226.github.io/GaussHDR/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  
    <a href='https://drive.google.com/file/d/1SZYoikKiCvBdGnWmJ1qHtGuspub_LkV4/view?usp=drive_link'><img src='https://img.shields.io/badge/Preprocessed-Data-blue'></a>  
    <a href='https://drive.google.com/file/d/1uaBfv_9boxl9pl3IMED5WIGcbsZMjUS9/view?usp=drive_link'><img src='https://img.shields.io/badge/Pretrained-Models-orange'></a> 
</div>
</p>

<br>


<table align="center">
  <tr>
    <td align="center">
      <video src="https://github.com/user-attachments/assets/4bb7cb6b-6d3a-412d-953f-0f182d7a35a9" controls width="100%"></video>
    </td>
    <td align="center">
      <video src="https://github.com/user-attachments/assets/224be397-e289-4326-8781-fabad4cc09e8" controls width="100%"></video>
    </td>
  </tr>
  <tr>
    <td align="center">
      <video src="https://github.com/user-attachments/assets/01d91f06-3e66-4083-b0e7-3f1ff0ca1b8f" controls width="100%"></video>
    </td>
    <td align="center">
      <video src="https://github.com/user-attachments/assets/47343c8b-903a-410e-9f59-f9a717a3864a" controls width="100%"></video>
    </td>
  </tr>
</table>



## Setup
### Clone the repo
```shell
git clone https://github.com/LiuJF1226/GaussHDR.git --recursive
cd GaussHDR
```
### Install dependencies
```shell
conda create -n gausshdr python=3.9
conda activate gausshdr

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.12.1+cu113.html

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install -r requirements.txt
```

## Data
The datasets employed in this project stem from [HDR-NeRF](https://github.com/xhuangcv/hdr-nerf) (real and synthetic) and [HDR-Plenoxels](https://github.com/kaist-ami/HDR-Plenoxels) (real). We preprocess the data through COLMAP to make it suitable for GS training. Please download our preprocessed data from [Google Drive](https://drive.google.com/file/d/1SZYoikKiCvBdGnWmJ1qHtGuspub_LkV4/view?usp=drive_link). Then, you can unzip it to any directory you like, for example in folder ```GaussHDR/datasets/```. The data structure will be organised as follows:

```
datasets/
├── HDR-NeRF-real/
│   ├── box/
│   │   ├── images
│   │   │   ├── xxxx.jpg
│   │   │   ├── xxxx.jpg
│   │   │   ├── ...
│   │   ├── sparse/
│   │   |   └──0/
|   |   └── exposure.json
│   ├── ...
│   ├── ...
├── HDR-NeRF-syn/  
│   ├── bathroom/
│   │   ├── images
│   │   │   ├── xxxx.png
│   │   │   ├── xxxx.png
│   │   │   ├── ...
│   │   ├── images_hdr
│   │   │   ├── xxxx.exr
│   │   │   ├── xxxx.exr
│   │   │   ├── ...
│   │   ├── sparse/
│   │   |   └──0/
|   |   └── exposure.json
│   ├── ...
│   ├── ...
├── HDR-Plenoxels-real/ 
│   ├── character/
│   │   ├── images
│   │   │   ├── xxxx.jpg
│   │   │   ├── xxxx.jpg
│   │   │   ├── ...
│   │   ├── sparse/
│   │   |   └──0/
|   |   └── exposure.json
│   ├── ...
│   ├── ...
```
## Training
### Training on a single scene
If you want to train on a single scene for a quick start, you can use following commands.
```shell
# HDR-NeRF-syn data (bathroom)
python train.py --gpu 0 -s datasets/HDR-NeRF-syn/bathroom/ -m logs_exp3/HDR-NeRF-syn/bathroom -r 2 -d synthetic --voxel_size 0.001 --update_init_factor 16 --gamma 0.5 --exp_mode 3 

# HDR-NeRF-real data (box)
python train.py --gpu 0 -s datasets/HDR-NeRF-real/box/ -m logs_exp3/HDR-NeRF-real/box -r 4 -d real --voxel_size 0.001 --update_init_factor 16 --gamma 0.2 --exp_mode 3 

# HDR-Plenoxels-real data (coffee)
python train.py --gpu 0 -s datasets/HDR-Plenoxels-real/coffee/ -m logs_exp3/HDR-Plenoxels-real/coffee -r 6 -d real --voxel_size 0.001 --update_init_factor 16 --gamma 0.2 --exp_mode 3 
```
- gpu: specify the GPU id to run the code.
- source_path (s): data path of the training scene.
- model_path (m): logging path.
- resolution (r): training resolution.
- data_type (d): training data type (synthetic or real).
- voxel_size: from [Scaffold-GS](https://github.com/city-super/Scaffold-GS), size for voxelizing the SfM points, smaller value denotes finer structure and higher overhead, '0' means using the median of each point's 1-NN distance as the voxel size.
- update_init_factor: from [Scaffold-GS](https://github.com/city-super/Scaffold-GS), initial resolution for growing new anchors. A larger one will start placing new anchor in a coarser resolution.
- gamma: the weight of global tone-mapping loss term.
- exp_mode: training exposure setting, to be 1 or 3. Exp-1 means only one exposure is used for each view during training (following HDR-NeRF), while Exp-3 means all three exposures are accessible during training (alighing with HDR-GS).

### Training on multiple scenes
We also provide the scripts to train on all the scenes at one time.
 - HDR-NeRF-syn: ```train_hdrnerf_syn.sh```
 - HDR-NeRF-real: ```train_hdrnerf_real.sh```
 - HDR-Plenoxels-real: ```train_hdrplenoxels_real.sh```

Run them with ```bash train_xxx.sh```.

> Notice: You can either use sequential training or parallel training by commenting out/uncomment corresponding parts in the ```train_xxx.sh``` files. Some hyper-parameters and training settings can be also modified in these files. For parallel training, make sure you have enough GPU cards and memories to run these scenes at the same time. 

## Evaluation

### Evaluation after training
 > Notice: Above training scripts will also generate LDR/HDR renderings, compute and log error metrics during training process. Therefore, you may ignore this part.

After training, you can generate rendering results, compute and log error metrics for a trained scene as follows. 
```shell
python render.py -m <logging path of a trained scene> --gpu 0 # Generate renderings
python metrics.py -m <logging path of a trained scene> --gpu 0 # Compute and log error metrics on renderings
```

### Evaluation on pretrained models
We also provide our pretrained [models](https://drive.google.com/file/d/1uaBfv_9boxl9pl3IMED5WIGcbsZMjUS9/view?usp=drive_link) that correspond to the metrics in paper, including ```exp1_models (exp_mode=1)``` and ```exp3_models (exp_mode=3)```. You can first download and unzip them to any directory. Then, run the following commands. 
```shell
python render.py -m <pretrained model path of a scene> -s <data path of this scene> --gpu 0 # Generate renderings
python metrics.py -m <pretrained model path of a scene> --gpu 0 # Compute and log error metrics on renderings
```


## Citation
```BibTeX
@inproceedings{gausshdr,
     title={GaussHDR: High Dynamic Range Gaussian Splatting via Learning Unified 3D and 2D Local Tone Mapping},
     author={Jinfeng Liu and Lingtong Kong and Bo Li and Dan Xu},
     booktitle={CVPR},
     year={2025}
     }
```
## Acknowledgement
This repo is mainly based on [Scaffold-GS](https://github.com/city-super/Scaffold-GS). We thank all authors from [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) and [Scaffold-GS](https://github.com/city-super/Scaffold-GS) for presenting such excellent works. 
