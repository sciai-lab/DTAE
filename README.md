# Density Tree-biased AutoEncoder (DTAE)

This is the official implementation of the paper : "Visualizing hierarchies in scRNA-seq data using a density tree-biased autoencoder"

---
## Requirements

All requirements are found in the file *requirements.txt*.

You will additionaly have to install **Speedrun**.
An installation guide is available at [github.com/inferno-pytorch/speedrun](https://github.com/inferno-pytorch/speedrun).

---
## Usage 

The training takes place in two parts, first the pretraining and then the finetuning.
We assume that the data is named `X.csv` and its folder is defined in the experiment's configuration.
In order to train a model, you can either create manually a configuration folder/files like in the folder **experiments** and then use the commands:

```
python <pretrain/finetune>.py <experiments_folder>/<experiment_name> 
```
Or you can use an already existing experiment and use :
```
python <pretrain/finetune>.py <experiments_folder>/<experiment_name> \
--inherit <experiments_folder>/<reference_experiment_name>
```
Notice that here that the folder `<experiment_name>` doesn't need to exist already
To modify this existing experiment you can use the command line parameters
```
--config.<config_element> <new_element_value>
```
As such, modifying the dimensions of the autoencoder in an experiment would look like
```
python <pretrain/finetune>.py <experiments_folder>/<experiment_name> \
--inherit <experiments_folder>/<reference_experiment_name> \
--config.model.kwargs.dims [<dim_1>,<dim_2>,...,<dim_n>]
```
For more details regarding Speedrun's usage, please refer to [github.com/inferno-pytorch/speedrun](https://github.com/inferno-pytorch/speedrun).

After the training, the final embedding is in the file `experiments/experiment/embedding.csv`. Embeddings are also stored periodically during training, and logs are visualizable during training using Tensorboard. They are found in the folder `experiments/experiment/Logs`.

---
## Reproducibilty 
We provide both template experiments and the configurations used to reproduce our results in folders `experiments/<experiment>-<pretraining/finetuning>`.

Pretrained networks used to produce the figures in the paper are available on [Google Drive](https://drive.google.com/drive/folders/1cmcTQyJzwg76JiaWhISFZIvOVX05qhTS?usp=sharing)

The preprocessed endocrine pancreas dataset is available at : [GSE 132188](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE132188) 

The raw dentage gyrus dataset is available at : [GSE 104323](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE104323)

---
## Example experiment

We will take a look at how to reproduce the experiment on the PHATE generated data.

The process is as follows :
```
python phate_data_generation.py
python pretrain.py experiments/phate-gen-pretraining --config.device cuda:0
python finetune.py experiments/phate-gen-finetuning --config.device cuda:0
python embedding_visualization.py --path experiments/phate-gen-finetuning
```

The first line creates the dataset and stores it in the folder `data`.

The second one will pretrain the DTAE on the cgenerated data(`data/X.csv`), on device `cuda:0`. Feel free to use the appropriate device here.

Then the third will finetune the DTAE and produce the final embedding.

The fourth line is here to visualize in a very simple way the embedding obtained during training.



