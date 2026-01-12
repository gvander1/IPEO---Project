### IPEO---Project
# Comparison of satellite images and learned embeddings for land cover mapping in the Brazilian Amazon

# 1. Install dependencies
In order to run the code following librairies need to be installed by running following lines in a terminal:
```bash
pip install numpy
pip install torch
pip install rasterio
pip install pandas # not sure if this is needed in the end...
pip install seaborn
```

# 2. Data preprocessing
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

# 3. Model training
Now let's train our 3 models !
## a. Deep Learning model (convnet) for Sentinel-2 semantic segmentation
For model training, first connect to the SCITAS GPU and activate a virtual environment
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

## b. Deep Learning model for AE Embeddings semantic segmentation
A GPU is also needed ! Follow the same instruction as for 3.a.
```bash
python train_model_AE_DL.py
```

## c. Random Forest for AE Embeddings
Work in progress

# 4. Visualisation
To access qualitatve assessment of the semantic segmentation, you can run a jupyter notebook to visualise
the satellite image in RGB together with groundtruth and model output. To do so, open a notebook with:
```bash
jupyer notebook plot.ipynb
```
