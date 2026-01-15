### IPEO---Project / EPFL - ENV-540
# Comparison of satellite images and learned embeddings for land cover mapping in the Brazilian Amazon

Van der Bruggen Gaétane, Steiner Maxime, Ramabadran Philip

The aim of this project was to train 3 different semantic segmentation models to map land cover in the Brazilian Amazon. The performance of one deep-learning convnet on Sentinel-2 imagery and two AlphaEarth Embedding-base models (a Random Forest per pixel Classifier and a deep-learning convnet) were compared. For further details please refer to the provided report.

# A. Repo Structure

```bash
.
├── Test_for_inference/        # Example data for running inference
├── archives_unused_scripts/   # Old experimental scripts (not used in main pipeline)
├── final_metrics/             # Stored metrics and evaluation results (reproducible)
├── mean_std/                  # Mean / std statistics for normalization
├── modeloutputs/              # Saved model predictions and intermediate outputs (not final !!)
│   ├── AE_accuracy/           # Different accuracies over Epochs during training
│   ├── AE_accuracy_final/      
│   ├── metrics/               # Confusion Matrices
│   └── s2_accuracy/            
├── models/                    
│   ├── AE/                    # former model used for fine-tune
│   ├── AE_final/              # former model
│   └── lcamazon.py            # Class that loads and preprocesses the data, called during training / central pipeline architecture
├── Images_to_discard.npy      # Indices of discarded images
├── Images_to_disregard.py     # Script defining images to ignore
├── acc_for_hp_RF              # Accuracy logs for RF hyperparameter choice
├── class_frequency.npy        # Class frequency for weighted Cross-Entropy
├── compute_mean_std.py        # Script to compute mean/std over the train/val dataset for normalization
├── environment.yml            # Conda environment specification
├── inference.ipynb            # Notebook to run inference
├── label_proportion.py        # Compute label proportions for weighted Cross-Entropy
├── metrics.py                 # Script to compute confusion matrices and save prediction after the models are trained
├── plot.ipynb                 # Notebook for plotting results after predictions are saved
├── train_model_AE_DL.py       # Training script for AE_DL model
├── train_model_AE_RF.py       # Training script for AE_RF model
├── train_model_sentinel.py    # Training script for Sentinel-2 DL model
├── utils.py                   # Useful functions called by other scripts
└── README.md                  # Project documentation --> the one your reading right now !
```
# B. Requirements

Install dependencies using the provided environment.yml file

```bash
environment.yml
```

A jupyter notebook is provided to run inference on 2 test images. To use it, you must download the weights of our trained models using [this link](https://filesender.switch.ch/filesender2/?s=download&token=143b661c-e4a8-4b7d-8024-6f09b56988e0)

The test images are already provided in this git in the directory "Test_for_inference". Nothing has to be done about it.

If you want to reproduce our results, you must download orginal DATA using [this link](https://filesender.switch.ch/filesender2/?s=download&token=1cebb60f-874d-4be1-9e88-82c1ade7439a)

It should look like this:
```bash
IPEO---PROJECT
├── DATA/                       # Downloaded DATA here
│   ├── S2/          
│   ├── labels/
│   ├── AE/
│   └── Codigos-da-legenda-colecao-9.csv
├── final_models/               # Downloaded trained model weights here
│   ├── s2_final.pth         
│   ├── AE_final.pth
│   └── AE_RF_final/
│       ├── rf_model.pkl
│       └── scaler.pkl
```

# C. Inference

In order to run inference, make sure the model weights are downloaded and put in the main directory, then run the jupyter notebook **inference.ipynb** by running the following line in the terminal:

```bash
jupyter notebook inference.ipynb
```
It will load some functions from the script **utils.py**

# D. Reproduce results

If you want to reproduce our results, first downloaded the "DATA" folder (cf. B.), then follow these instructions.

## 1. Data preprocessing
Since the original data has some classes that only appear in one of the test/train set, we need to disregard 483
images that contain those classes when loading the data. This has to be done by running the script **Images_to_disregard.py**
```bash
python Images_to_disregard.py
```
Then, for later purposes, we need to compute the mean and standard deviation over each 12 bands of sentinel images and
each 64 layers of AE Embeddings. This is achieved by running the script **compute_mean_std.py**
```bash
python compute_mean_std.py
```
Then also compute the proportion of each label on the dataset for later purposes (e.g. dealing with class imbalance).
To do so, run the script **label_proportion.py**
```bash
python label_proportion.py
```

## 2. Model training
Now let's train our 3 models !
### a. Deep Learning model (convnet) for Sentinel-2 semantic segmentation
For model training, first connect to the SCITAS GPU and activate a virtual environment. It must have all the dependencies listed in environment.yml
```bash
Sinteract -a env540 -g gpu:1 -t 03:00:00
source ipeo_venv/bin/activate
```
Then go back to the working directory **IPEO---PROJECT** using the cd command
```bash
cd IPEO---PROJECT
```
Now you can run **train_model_sentinel.py** using following line:
```bash
python train_model_sentinel.py
```
If the terminal prints **GPU to be started**, this means the GPU is not available. If this is the case
please try again the first steps to connect to the GPU

The trained model will be saved.
### b. Deep Learning model for AE Embeddings semantic segmentation
A GPU is also needed ! Follow the same instruction as for 2.a.
```bash
python train_model_AE_DL.py
```
The trained model will be saved.
### c. Random Forest for AE Embeddings
Run this line in terminal:
```bash
python train_model_AE_RF.py
```
The trained model will be saved.
## 3. Computing metrics
Compute metrics by running this script:
```bash
python metrics.py
```
Confusion matrices will be saved in the **final_predictions** directory
Prediction maps will be saved in the **final_predictions** directory

## 4. Visualisation
To access qualitatve assessment of the semantic segmentation, you can run a jupyter notebook to visualise
the satellite image in RGB together with groundtruth and model output. To do so, open a notebook with:
```bash
jupyer notebook plot.ipynb
```


And that's it !