# Contrastive Learning for Electronic Health Record Data  

## Data
- We used MIMIC database, and in order to compare to others' works, we adopted the benchmark process on this data. 
- The folder `mimic3benchmark` and `mimic3models` are forked from https://github.com/YerevaNN/mimic3-benchmarks.

## Environment


### Pytorch
We used Pytorch to write the contrastive learning code. The core code is saved in the `models` folder.
### fastai
We used fastai to do the downstream prediction tasks. 


## Structure
### Different settings of the networks
To enable different settings (different hyper-parameters and other parameters for training) easily used and recorded automatically, we have a folder `logs`. In the `logs`, write a setting as `setting1.ini` /`setting2.ini`, then run the script in the 'scripts' folder with this setting. 

### functions and models 
We put the functions and models in the `models` folder.

### scripts
We put the major scripts in the folder `scripts`.

