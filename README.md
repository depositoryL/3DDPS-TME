# 3DDPS-TME Source Code

## I. Structure

- Configs folder: All adjustable parameters are stored in these yaml files, just change them if necessary;

- Data folder: The csv files containing the network traffic and routing matrix;

- DDPM folder: The diffusion model code;

- running_commands folder: The execution of internal files to complete the main method of each scenario;

- Utils folder: The relevant code for pre-processing, post-processing, accessing parameters, reading and writing the log.

- train_run.py: The basic template of the main method;

## 2. The process of using

### 1. First, use __win + r__ to open the __cmd__ terminal;

### 2. Then copy the absolute path of the project and type it into the terminal:

```bash
cd C:\absolute_path\3DDPS-TME
```

### 3. Finally, run the program in the running_commands folder (using the Abilene dataset as an example, still in the terminal):

 ```bash
python running_commands/train_run_dm_abilene.py
```

## 3. Cautions

### Changing the program to execute different targets

Only change the string variable in the py file corresponding to the running_commands folder:

 - Training:
 'python train_run.py --name abilene --config_file Configs/dm_abilene.yaml --gpu 0 --is_train'

 - Generate data:
 'python train_run.py --name abilene --config_file Configs/dm_abilene.yaml --gpu 0 --is_sample'

 - Traffic estimation:
 'python train_run.py --name abilene --config_file Configs/dm_abilene.yaml --gpu 0'

### 2. Network Traffic Trimming

Since the measured traffic will have abnormal (large) values, resulting in poor normalisation (the vast majority of values will be less than 0.001). One option is to perform a smoothing operation on these extreme flows, and the code (Utils/data_utils.py) can optionally remove the above values according to the 99.5th quartile (you can set this yourself). Trimming is not used by default, but if you need it, just add --clip to the end of the string variable, e.g:

- python train_run.py --name abilene --config_file Configs/dm_abilene.yaml --gpu 0 --is_train ---clip'.

### 3. About the folder OUTPUT

At the start of training, the OUTPUT folder is immediately created to store some future output files, including estimated traffic, generated traffic, real traffic, model parameter records and log files. When you have finished running the program and need to process it further (draw graphs and check accuracy), remember to look for the relevant files in this folder.

### 4. About log files

Will not cover the same operation last time, only then write the log, if many times debugging, remember to delete the previous log file

Translated with DeepL.com (free version)