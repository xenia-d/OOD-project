# A Comparison of Out-of-Distribution (OOD) Detection Methods for Near and Far OOD

The code in this repository is the implementation for the project titled "Comparing OOD Detection Methods on Near and Far OOD" for the course "Trusthworthy and Explainable AI". In safety critical applications, machine learning models should be able to detect encounters of unknown classes to alert users to conduct human evaluations. Such samples are referred to as "out-of-distribution" (OOD) and significantluy deviate from the training set of a machine learning model. However, identifying OOD samples come at various difficulties which can be broadly categorized into "Near-OOD" (the OOD sample is either perceptually or semantically dissimilar to the inlier classes) or "Far-OOD" (the OOD sample is both perceptually and semantically dissimilar to the inlier classes).
In this project we implement 5 uncertainty quantification-grounded methods for OOD detection: Maximum Confidence, Shannon Entropy, Ensembles, ODIN and DUQ to address the following research questions:

1. How do various out-of-distribution (OOD) methods compare in terms of (Near and Far) OOD detection abilities?
2. How does Far OOD detection performance compare to Near OOD detection?

We implement 2 experiments;A basic and an advanced experiment. Both experiments train a model on an in-distribution class (ID) and evaluate OOD detection on a Near-OOD and Far-OOD dataset. Our results show that DUQ has consistently detected OOD well for both Near and Far-OOD, while ODIN has perfomed relatively poorly on both Near and Far-OOD, across both experiments. 

## Acknowledgements
Below is a overview of where code was used directly or as inspiration from other sources.
- The code for the ResNet18 was taken from "https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py", this is also mentioned in its folder *Model_Architecture/Adv_CNN.py*.
- The code for the MNIST model was inspired from "https://github.com/pytorch/examples/blob/main/mnist/main.py", mentioned in *Model_Architecture/baseline_CNN.py*.
- DUQ specific code was adapted from "https://github.com/y0ast/deterministic-uncertainty-quantification", it was used as a base, but altered to match our experimentation.
- ODIN specific code was adapted from "https://github.com/facebookresearch/odin", again it was used as a base, but altered to match our experimentation.

## General Setup
Please setup a venv and install the requirements.txt.
All our trained models can be found under Saved Models.

## Repository Overview
All the main files for running experiments and plotting results are directly under the root. Below is an overview of each folder.
- *Data* contains classes for each dataset and is where the datasets will download. 
- *DUQ* contains code for running DUQ.
- *Model_Architecture* contains the model architectures for the MNIST model and ResNet18 model.
- *ODIN* contains code for running ODIN.
- *Saved Images* contains a sample of input images from each dataset.
- *Saved Models* contains the saved models for MNIST, CIFAR10 and DUQ.
- *Saved Plots* contains plots for each method on each dataset and the *Overall* plots comparing all method and datasets
- *Saved Rocks* contains the saved TPR and FPR from the ROC curve for each ID/OOD dataset pair and method.
- *Train* contains the scripts to train the baseline models for MNIST and CIFAR10
- *Utils* contains extra auxiliary functions

## Running Experiments
The results of all experiments are already saved. This is done in the form of saving the TPR and FPR from the ROC plot for each method and each pair of in-distribution (ID) and OOD datasets. These results can be found under the *Saved Rocks* folder, where each method has a folder, and within there is a folder for the basic experiment (with MNIST as ID) and the advanced experiment (with CIFAR10 as ID). Where applicable the results of the 5 trials are saved.

For a quick run the plotting can be directly run from the saved results as explained below in **Plotting**.

To rerun the experiments each method is explained below.

### Model Training
Trained models are already saved, but a new model can be trained as explained below. This will automatically download the respecitve dataset, and it may take some time to train.

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

### DUQ Experiments 
Contrary to the above methods, DUQ does not require baseline models to be trained. This method instead trains a feature extractor based on the baseline model architectures (without the final layers) to obtain learnable centroids for the method. For the  MNIST and CIFAR experiments, these have separate main files to run. To run the DUQ experiments:

for the MNIST experiment:

```
python main_DUQ_MNIST.py
```

For the CIFAR10 experiment it is rather computationally expensive. If running on your local computer it can be run with:

```
python main_DUQ_CIFAR.py
```

However, we recommmend running the job script "job_DUQ_advanced.py" on Habrok, as we have done for this project. 

Performing OOD detection for the CIFAR experiment is done in a separate file, under the folder "DUQ", adn assumes you have pretrained models of the DUQ method saved in the "Saved Models" folder. Once you are in the DUQ folder, the OOD detection can  be run with the following command:

```
python perform_OOD_CIFAR.py
```


## Plotting
Plots (such as ROC comparisons and bar plot of average AUROCs) can be achieved using the functions in **final_plotting.py**.