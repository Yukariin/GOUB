<h1 align="center"> Image Restoration Through Generalized Ornstein-Uhlenbeck Bridge </h1>

Official PyTorch Implementations of GOUB, a diffusion bridge model that applies the Doob's *h*-transform to the generalized Ornstein-Uhlenbeck process. This model can address general image restoration tasks without the need for specific prior knowledge.

# Overview
<div align="center">
    <img src="figs/framework.png" alt="Framework">
</div>

# Visual Results
<div align="center">
    <img src="figs/ir.png" alt="Framework" width="60%"><br>
</div>

# Intallation
This code is developed with Python3, and we recommend python>=3.8 and PyTorch ==1.13.0. Install the dependencies with Anaconda and activate the environment with:

    conda create --name GOUB python=3.8
    conda activate GOUB
    pip install -r requirements.txt

# Test
1. Prepare datasets.
2. Download pretrained checkpoints here.
3. Modify options, including dataroot_GT, dataroot_LQ and pretrain_model_G.
4. Choose a model to sample (Default: GOUB): test function in `codes/models/denoising_model.py`.
5. `python test.py -opt=options/test.yml`

The Test results will be saved in `\results`.

# Train
1. Prepare datasets.
2. Modify options, including dataroot_GT, dataroot_LQ.
3. `python train.py -opt=options/train.yml` for single GPU.<br> `python -m torch.distributed.launch --nproc_per_node=2 --master_port=1111 train.py -opt=options/train.yml --launcher pytorch` for multi GPUs. *Attention: see [Important Option Details](#important-option-details)*.

The Training log will be saved in `\experiments`.

# Important Option Details
* `dataroot_GT`: Ground Truth (High-Quality) data path.
* `dataroot_LQ`: Low-Quality data path.
* `pretrain_model_G`: Pretraind model path.
* `GT_size, LQ_size`: Size of the data cropped during training.
* `niter`: Total training iterations.
* `val_freq`: Frequency of validation during training.
* `save_checkpoint_freq`: Frequency of saving checkpoint during training.
* `gpu_ids`: In multi-GPU training, GPU ids are separated by commas in multi-gpu training.
* `batch_size`: In multi-GPU training, must satisfy relation: *batch_size/num_gpu>1*.

# FID
We provid a brief guidelines for commputing FID of two set of images:

1. Install FID library: `pip install pytorch-fid`.
2. Commpute FID: `python -m pytorch_fid GT_images_file_path generated_images_file_path --batch-size 1`<br>if all the images are the same size, you can remove `--batch-size 1` to accelerate commputing.
