### IPEO---Project / EPFL - ENV-540
# Comparison of satellite images and learned embeddings for land cover mapping in the Brazilian Amazon

Van der Bruggen Gaétane, Steiner Maxime, Ramabadran Philip

The aim of this project was to train 3 different semantic segmentation models to map land cover in the Brazilian Amazon. The performance of one deep-learning convnet on Sentinel-2 imagery and two AlphaEarth Embedding-base models (a Random Forest per pixel Classifier and a deep-learning convnet) were compared.

# A. Requirements

Install dependencies using the provided environment.yml file

```bash
environment.yml
```

A jupyter notebook is provided to run inference on 2 test images. To use it, you must download the weights of our trained models using [this link](https://filesender.switch.ch/filesender2/?s=download&token=143b661c-e4a8-4b7d-8024-6f09b56988e0)

The test images are already provided in this git in the directory "Test_for_inference". Nothing has to be done about it.

If you want to download orginal DATA to reproduce our results, together with the prediction maps of our models and performance metrics, please use [this link]

# B. Inference

In order to run inference, make sure the model weights are downloaded and put in the main directory, then run the jupyter notebook **inference.ipynb** by running the following line in the terminal:

```bash
jupyter notebook inference.ipynb
```
It will load some functions from the script **utils.py**

# C. Reproduce results

If you want to reproduce our results, first downloaded the "DATA" folder (cf. A.), then follow these instruction.

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
A GPU is also needed ! Follow the same instruction as for 3.a.
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
Run this line in terminal:
```bash
python metrics.py
```
Confusion matrices will be saved in the **final_predictions** directory

# 4. Visualisation
To access qualitatve assessment of the semantic segmentation, you can run a jupyter notebook to visualise
the satellite image in RGB together with groundtruth and model output. To do so, open a notebook with:
```bash
jupyer notebook plot.ipynb
```
