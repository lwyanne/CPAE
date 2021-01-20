# CPAE

## Data
- We used MIMIC database, and in order to compare to others' works, we adopted the benchmark process on this data. 
- The folder `mimic3benchmark` and `mimic3models` are forked from [**YerevaNN#mimic3-benchmarks**](https://github.com/YerevaNN/mimic3-benchmarks).
 
## Data Preprocessing
- Following guidance in the section [**Building benmark**](https://github.com/YerevaNN/mimic3-benchmarks#building-a-benchmark), the raw timeseries data will be extracted in a `csv` format.  
    ```python
    # 1 seperate patient from pool
    python -m mimic3benchmark.scripts.extract_subjects {PATH TO MIMIC-III CSVs} data/root/
    # 2 filtering low quality event
    python -m mimic3benchmark.scripts.validate_events data/root/
    # 3 seperate different episodes from the same patient
    python -m mimic3benchmark.scripts.extract_episodes_from_subjects data/root/
    # 4 split  train / test 
    python -m mimic3benchmark.scripts.split_train_and_test data/root/

    # 5 generate downstream task data
    python -m mimic3benchmark.scripts.create_in_hospital_mortality data/root/ data/in-hospital-mortality/
    python -m mimic3benchmark.scripts.create_length_of_stay data/root/ data/length-of-stay/
    ```
    > The default setting used by [3.episode extraction](https://github.com/YerevaNN/mimic3-benchmarks/blob/master/mimic3benchmark/scripts/extract_episodes_from_subjects.py) applied a strong selection of feature left for later analysis.   --seed 1 
    > We provide an new API to allow a more relax feature selection. Slosling save your list of `ITEMID` into a txt file with `--seed 1 \t` selength-of-stayto the script via `--variables_to_
    
    ```seed number (used to sample subset) and the ini-file (possible choises are in /home/logs/los/FineTune for Length-of-Stay and /home/logs/imp/FineTune/ for In-hospital mortality.). The scripts will directly load models from the ini-file and all other needed parameters.
    Note that you need to specify the keep`.
    ```python
    # [optional] 3 sepearte episodes while keeping ITEM of interests 
    python -m mimic3benchmark.scripts.extract_episodes_from_subjects data/root/ --variables_to_keep {abs path}/My_desire_ITEM.txt     
    ```
- Further preprocessing includes imputation and normalization. These two steps are inset into the `datareader`
    - Imputation : Four strategy : `['zero', 'normal_value', 'previous', 'next']`: we build our performance comparison based on strategy 'previous'.  
    - Normalization : `./ref/ihm_ts:0.25_impute:previous_start:zero_masks:True_n:17903.normalizer` recorded the normalization setting and it's called defaultly . 




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

## How to Use
1. Complete the `machineConfig.json`. This json file allow you to put the `data/` and `logs/` at different directory.
e.g. 
```json
{
    "script_dir": "/home/CPAE/",
    "data_dir": "/home/CPAE/data",
    "logging_dir": "/home/CPAE/logs"
  }
```
2. To pretrain a model:
``` Shell
/home/CPAE/scripts/main_pretrain.py --ini-file {model_dir}/{setting_name}.ini --resume False
```
The code will automatically instance a model according to the {model_dir} and the setting file. Please see below to choose {model_dir} and parameters in setting.ini:
```
Model      model_dir       sample ini file
AE_l       AE_LSTM            st.ini
CAE_l      CAE_LSTM           st.ini
CPC_l      CPAELSTM44         cpc.ini
BaseCPAE_l CPAELSTM44         st.ini
AtCPAE_l   CPAELSTM44_AT      st.ini
AE_c       AE2                st.ini
CAE_c      CAE2               st.ini
CPC_c      CPAE1              cpc.ini
BaseCPAE_c CPAE1              st.ini
AtCPAE_c   CPAE_AT            st.ini
```

3. Downstream tasks
Note that the pretrained models are available in corresponding `model_dir`. To perform downstream prediction, use the following command:
```
/home/CPAE/scripts/main_finetune_imp.py --seed 1 --ini-file /home/logs/imp/FineTune/sup_lstm.ini # in hospital mortality prediction
/home/CPAE/scripts/main_finetune_los.py --seed 1 --ini-file /home/logs/los/FineTune/sup_lstm.ini # length-of-stay prediction

```
Note that you need to specify the seed number (used to sample subset) and the ini-file (possible choises are in /home/logs/los/FineTune for Length-of-Stay and /home/logs/imp/FineTune/ for In-hospital mortality.). The scripts will directly load models from the ini-file and all other needed parameters.