# A Comparison of Out-of-Distribution (OOD) Detection Methods for Near and Far OOD

add paper abstract here

## General Setup
Please setup a venv and install the requirements.txt.
All our trained models can be found under Saved Models.

## Running Experiments
The results of all experiments are already saved. This is done in the form of saving the TPR and FPR from the ROC plot for each method and each pair of in-distribution (ID) and OOD datasets. These results can be found under the *Saved Rocks* folder, where each method has a folder, and within there is a folder for the basic experiment (with MNIST as ID) and the advanced experiment (with CIFAR10 as ID). Where applicable the results of the 5 trials are saved.

For a quick run the plotting can be directly run from the saved results as explained below in **Plotting**.

To rerun the experiments each method is explained below.

### Model Training
Trained models are already saved, but a new model can be trained as explained below.

For MNIST:
```
python -m Train.train_mnist_model --batch_size 64 --epochs 5 --lr 0.001 --model_number 1
```
And for the CIFAR10:
```
python -m Train.train_cifar_model --batch_size 64 --epochs 5 --lr 0.001
```

All arguments shown are already the default so if no alterations need to be made then they do not need to be specifided. An additional note, the CIFAR model trainer trains 5 models in a row, so no model_number is needed for saving.

### ODIN Experiments
This requires the baseline models (for either MNIST or CIFAR10, depending on which one you want to run the experiment on) to be trained. To run ODIN experiments 
for the MNIST experiment:

```
python main_ODIN.py --nn BASELINE_CNN --model_num all --magnitude 0 --temperature 1000
```
And for the CIFAR10 experiment:
```
python main_ODIN.py --nn ADVANCED_CNN --model_num all --magnitude 0.0014 --temperature 1000
```

Note that the **--device** and **--batch_size** can be specified as well. Results will be saved under Saved_Rocks.

### Ensemble & Entropy Experiments
This also requires the baseline models on MNIST and CIFAR10 to be trained. This file assumes that all 5 baseline models for each dataset have been trained as 5 iterations of the baseline entropy OOD detection are performed on each dataset (and its corresponding OOD datasets) as well as for the ensemble. It can be run as shown below.

```
python main_ensemble_and_entropy.py
```

## Plotting
Plots (such as ROC comparisons and bar plot of average AUROCs) can be achieved using the functions in **final_plotting.py**.