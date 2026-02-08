# GraMIo: Mutigating Position Bias in Inductive Spatiotemporal Learning via Multimodal Graphs
This repository contains the codebase for GraMIo.

## Requirements
Main package requirements:
- ```Python == 3.10```
- ```PyTorch == 2.1.0```
- ```PyTorch Geometric == 2.6.1```
  - ```pyg_lib == 0.3.1```
  - ```torch_scatter == 2.1.2```
  - ```torch_sparse == 0.6.18```
  - ```torch_cluster == 1.6.3```
  - ```torch_spline_conv == 1.2.2```
- ```CUDA == 12.1```
- Torch Spatiotemporal Library (TSL): [https://github.com/TorchSpatiotemporal/tsl]

Install other packages through a conda environment using the following command from the root directory of the repository.

```
conda env create -f environment.yml
```

## Dataset
We use the following datasets:
- CARB-LA [https://pems.dot.ca.gov/]
- CARB-SF [https://pems.dot.ca.gov/]
- MADRID [https://www.madrid.es/portal/site/munimadrid]
- Synthetic

The datasets we use are available in ```/data```.

## Usage

All hyperparameter and dataset configurations are located in ```\model\config```. Specifically:
- ```\model\config\default-gramio.yaml``` contains general training configurations and hyperparameters.
- ```\model\config\dataset\*.yaml``` contains specific dataset configurations.
- ```\model\config\model\gramio.yaml``` contains specific GraMIo hyperparameters.

To train GraMIo on all datasets, run the following from the root directory.
```
bash run_exp-gramio.bash
```

To train GraMIo on a single run, an example for CARB-SF is provided below. Run the following from the root directory.
```
python3 model/mult_experiments.py '+seed=15900376' 'device=[0]' 'dataset=carb_sf' 'model.hparams.tra_sample_ratio=0.75' 'model.hparams.k=2' 'model.hparams.psd_layers=1' 'model.hparams.gcn_layers=2' 'optimizer.hparams.lr=0.001' 'model.regs.y1=0.1' 'model.regs.y2=0.0001' --config-name=default-gramio
```

Saved models are in ```/logs```, while csv results are in ```/res```.

To test a saved GraMIo model, run the following from the root directory.
```
python3 model/test_exp.py --config=<config_path.yaml> --checkpoint=<checkpoint_path.ckpt>
```

Trained model configs and checkpoint paths are located in ```/logs```.
Training results are located in ```/res```.
