# 4DVarNet

Implementation of the 4DVarNet framework for underwater acoustic reconstruction.

## Prerequisite

- git
- conda

## Install

### Install project dependencies

```
git clone https://github.com/CIA-Oceanix/4dvarnet-starter.git
cd 4dvarnet-starter
conda install -c conda-forge mamba
conda create -n 4dvarnet-starter
conda activate 4dvarnet-starter
mamba env update -f environment.yaml
```

## Run the ECS reconstruction

The model uses hydra see [#useful-links].
To run the configuration for the ECS reconstruction

```
python main.py xp=sound_base 
```

### Gulfstream training

## eNATL

Here is an example of the reconstruction of the ECS using a 0.1 sampling rate (i.e. 10% of the data) for the eNATL dataset.

|     |  ECS Metrics |
|:----|-------------:|
| μ   |   0.38142    |
| σ   |   0.14853    |
| λx  |   0.802      |
| λt  |   8.308      |

Animation:
![Animation](https://s3.eu-central-1.wasabisys.com/melody/eNATL/ECS/animation_git.gif)

## eNATL training and NATL testing

To run the configuration for the ECS reconstruction of the NATL dataset using the eNATL dataset as training.

```
python main.py xp=sound_base_transfert 
```

Here is an example of the reconstruction of the ECS using a 00.1 sampling rate (i.e. 1% of the data) of the NATL dataset using the eNATL dataset as training.

|    |   Metrics |
|:---|----------:|
| μ  |   0.6228  |
| σ  |   0.32826 |
| λx |   3.223   |
| λt |   49.806  |

Animation:
![Animation](https://s3.eu-central-1.wasabisys.com/melody/eNATL/ECS/new_reco_NATL.gif)

## Useful links:

- [Hydra documentation](https://hydra.cc/docs/intro/)
- [Pytorch lightning documentation](https://pytorch-lightning.readthedocs.io/en/stable/index.html#get-started)
- 4DVarNet papers:
  - Fablet, R.; Amar, M. M.; Febvre, Q.; Beauchamp, M.; Chapron, B. END-TO-END PHYSICS-INFORMED REPRESENTATION LEARNING FOR SA℡LITE OCEAN REMOTE SENSING DATA: APPLICATIONS TO SATELLITE ALTIMETRY AND SEA SURFACE CURRENTS. ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences 2021, V-3–2021, 295–302. <https://doi.org/10.5194/isprs-annals-v-3-2021-295-2021>.
  - Fablet, R.; Chapron, B.; Drumetz, L.; Mmin, E.; Pannekoucke, O.; Rousseau, F. Learning Variational Data Assimilation Models and Solvers. Journal of Advances in Modeling Earth Systems n/a (n/a), e2021MS002572. <https://doi.org/10.1029/2021MS002572>.
  - Fablet, R.; Beauchamp, M.; Drumetz, L.; Rousseau, F. Joint Interpolation and Representation Learning for Irregularly Sampled Satellite-Derived Geophysical Fields. Frontiers in Applied Mathematics and Statistics 2021, 7. <https://doi.org/10.3389/fams.2021.655224>.