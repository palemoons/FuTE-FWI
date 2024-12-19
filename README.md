# Towards complex seismic layers: A vision Transformer approach to full waveform inversion

This repository contains the source code for Fusion vision Transformer Enhanced network for Full Waveform Inversion (FuTE-FWI). 

## Enviroment Requirement

```text
python = 3.10.14
pytorch = 2.3.0
pytorch_msssim = 1.0.0
lpips = 0.1.4
deepwave = 0.0.20
numpy
einops
matplotlib
tqdm
```

Suggest using anaconda to manage the environment: 

```bash
conda create -n futefwi python=3.10
```

## Data Preparation

### OpenFWI

First, download model data from [OpenFWI official repository](https://github.com/lanl/OpenFWI). In our experiment, FlatVel-A, CurveVel-A, FlatFault-A and CurveFault-A dataset are required.

### OpenFWI-140

To generate any dataset of OpenFWI-140, use `gen_dataset.py`. Here is an example for generating FlatVel-D dataset:

```bash
python gen_dataset.py -d FlatVel -v A
```

The dataset splitting strategy is hard-coded in `dataset.py`. Modify this file as needed. The default strategy is as follows: 

| Dataset      | Train / test Split | Corresponding `.npy` files                  |
| ------------ | ------------------ | ------------------------------------------- |
| Vel Family   | 24k / 6k           | data(model)1-48.npy / data(model)49-60.npy  |
| Fault Family | 48k / 6k           | data(model)1-96.npy / data(model)97-108.npy |

## Training

To train FuTE-FWI, VelocityGAN and InversionNet, use the training script respectively. For example, to train FuTE-FWI on FlatVel-D, run the following command: 

```bash
python train_futefwi.py -d FlatVel -v A
```

Refer to the help text if you want to specify the default parameters.

## Testing

To evaluate the models, use `test.py`. The model and dataset name is required.

```bash
python test.py -m FuteFWI -d FlatVel -v A; % Test FuTE-FWI on FlatVel-D
python test.py -m VelocityGAN -d CurveVel -v A; % Test VelocityGAN on CurveVel-D
python test.py -m InversionNet -d FlatFault -v A --draw; % Test InversionNet on FlatFault-D. Add --draw if you need visualization
```

