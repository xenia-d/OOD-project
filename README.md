# A Comparison of Out-of-Distribution (OOD) Detection Methods for Near and Far OOD

add paper abstract here

## General Setup
Please setup a venv and install the requirements.txt.
All our trained models can be found under Saved Models.

## ODIN Experiments
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

## Plotting
Plots (such as ROC comparisons and bar plot of average AUROCs) can be achieved using the functions in **final_plotting.py**.