import matplotlib.pyplot as plt
import numpy as np
import warnings
import os
from skimage.io import imsave, imread
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib  # pip install joblib if not already there
from models.lcamazon import LCAmazon
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler
from skimage.segmentation import slic

# Ignore warnings form skimage
warnings.filterwarnings("ignore", message=".*low contrast image.*")

# Non deep-learning approach for semantic segmentation of the AE embeddings
# We first run a SLIC algorithm to get superpixels/regions
# We then train a random forest classifier on those regions for final segmentation
# Inspired by ex4 from the IPEO class

# Loading images
train_dataset = LCAmazon(root="DATA", modality="AE", split="train")
val_dataset = LCAmazon(root="DATA", modality="AE", split="val")

# Functions from ex4:
def convert_to_shape_pixels_by_bands(data):
    num_dimensions = len(data.shape)
    assert(num_dimensions == 2 or num_dimensions == 3)
    if num_dimensions == 3:
        num_bands = data.shape[2]
        return data.reshape((-1, num_bands))
    else:
        return data

def compute_average_feature(data):
    # If needed convert data to the shape (num_pixels x num_bands)
    data_2d = convert_to_shape_pixels_by_bands(data)
    avg_features = np.mean(data_2d, axis=0)
    return avg_features

def compute_standard_deviation_feature(data):
    # If needed convert data to the shape (num_pixels x num_bands)
    data_2d = convert_to_shape_pixels_by_bands(data)
    avg_features = np.std(data_2d, axis=0)
    return avg_features

def compute_histogram_feature(data, num_bins=10):
    # If needed convert data to the shape (num_pixels x num_bands)
    data_2d = convert_to_shape_pixels_by_bands(data)
    num_bands = data_2d.shape[1]
    hist_features = np.zeros((num_bands, num_bins)).astype(np.float32)
    for b in range(num_bands):
        # Compute the histogram for each band 
        #       use the function np.histogram(array, bins=num_bins)
        hist, boundaries = np.histogram(data_2d[:, b], bins=num_bins)
        hist_features[b, :] = hist
    # Return a 1D array containing all the values
    return hist_features.flatten()

def compute_image_features_from_regions(image, segmentation_map):
    region_id_list = np.unique(segmentation_map)
    all_features = []
    for region_id in region_id_list:
        # Obtain pixel values of each regions, with shape (num_pixels x num_bands)
        pixel_values = image[segmentation_map==region_id]
        # Compute the average, standard deviation and histogram features
        #       and concatenated them unsing the function (np.concatenate)
        avg = compute_average_feature(pixel_values)
        features = compute_standard_deviation_feature(pixel_values)
        hist_features = compute_histogram_feature(pixel_values)
        features = np.concatenate([avg, features, hist_features])
        # Add concatenated features to the variable all_features
        all_features.append(features)
    # convert list to numpy array of shape: (num_regions x num_bands)
    return np.array(all_features).astype(np.float32)

# Directory to save segmented images
os.makedirs('DATA/regions/', exist_ok=True)
os.makedirs('DATA/features/', exist_ok=True)

# Function for SLIC segmentation
def run_slic(dataset):
    print("------------------- Segmenting images with a SLIC algorithm and computing features -----------------------")
    for idx in tqdm(range(len(dataset))):
        image, _ = dataset[idx]
        # Recover original file name to save segmented image and features in the DATA folder
        _, gt_path = dataset.samples[idx]
        fname = os.path.basename(gt_path)
        regions_path = os.path.join("DATA/regions/", fname)
        region_features_path = os.path.join("DATA/features", fname).replace(".tif",".npy")
        # Segment image using SLIC
        segmented_image = slic(image, n_segments=30)
        # Compute features with functions defined above
        region_features = compute_image_features_from_regions(image, segmented_image)
        # Save features and segmented image
        imsave(regions_path, segmented_image.astype(np.uint32))
        np.save(region_features_path, region_features)


# Function from ex4
def get_label_per_region(segmented_image, label_map):
    """
    Returns a 1D numpy array that contains the label for each region, shape: (num_regions)
            For each region, we obtain the label that has the largest intersection with it
    """
    region_id_list = np.unique(segmented_image)
    label_id_list = np.unique(label_map)
    region_labels = []
    for region_id in region_id_list:
        mask_region = segmented_image == region_id
        
        intersection_per_label = []
        for label_id in label_id_list:
            mask_label = label_map == label_id
            # Compute intersection of each region with each label
            intersection = np.sum(mask_region * mask_label)
            intersection_per_label.append(intersection)
        
        intersection_per_label = np.array(intersection_per_label)
        # Obtain the index of the label with largest intersection
        selected_label = np.argmax(intersection_per_label)
        region_labels.append(selected_label)
    
    return np.array(region_labels).astype(np.uint32)

# Create arrays of training targets and features
print("----------- Loading region features and labels -----------")
all_train_region_features = []
all_train_region_labels = []
for idx in tqdm(range(len(train_dataset))):
    # Load image + ground truth from dataset
    image, gt = train_dataset[idx]
    # Recover file name of segmented image and features
    _, gt_path = train_dataset.samples[idx]
    fname = os.path.basename(gt_path)
    segmented_image_path = os.path.join("DATA/regions", fname)
    region_features_path = os.path.join("DATA/features", fname.replace(".tif", ".npy"))
    # Load segmented image and features
    segmented_image = imread(segmented_image_path)
    region_features = np.load(region_features_path)
    # Get labels per region using the function "get_label_per_region" defined above
    region_labels = get_label_per_region(segmented_image, gt)
    # Add current region labels to the variable all_train_region_labels
    all_train_region_labels.append(region_labels)
    # Add current region features to the variable all_train_region_features
    all_train_region_features.append(region_features)

# Tranforming the list all_train_region_labels in an array of shape: (num_all_regions)
train_labels = np.concatenate(all_train_region_labels)
# Tranforming the list all_train_region_features in an array of shape: (num_all_regions, num_features)
train_features = np.concatenate(all_train_region_features)

# Normalize features
mean_per_feature = np.mean(train_features, axis=0)
std_per_feature = np.std(train_features, axis=0)
norm_train_features = (train_features - mean_per_feature) / std_per_feature

# Train random forest classifier
classifier = RandomForestClassifier(random_state=10, n_estimators=2, max_depth=2)
classifier.fit(norm_train_features, train_labels)

# Predict classification maps for val images
print("--------------- Generating prediction maps for validation set ----------------")

for idx in tqdm(range(len(val_dataset))):

    # Load image + ground truth (GT not needed for prediction)
    image, _ = val_dataset[idx]

    # Recover filename from dataset metadata
    _, gt_path = val_dataset.samples[idx]
    fname = os.path.basename(gt_path)

    # Paths to segmented image + region features
    segmented_image_path = os.path.join("DATA/regions", fname)
    region_features_path = os.path.join("DATA/features", fname.replace(".tif", ".npy"))

    # Load segmented image
    segmented_image = imread(segmented_image_path)

    # Load region features
    region_features = np.load(region_features_path)

    # Normalize using training-set statistics
    norm_region_features = (region_features - mean_per_feature) / std_per_feature

    # Predict region labels
    label_predictions = classifier.predict(norm_region_features)

    # Build prediction map
    prediction_map = np.zeros(segmented_image.shape, dtype=np.uint8)
    region_id_list = np.unique(segmented_image)

    for ridx, region_id in enumerate(region_id_list):
        prediction_map[segmented_image == region_id] = label_predictions[ridx]

    # Save prediction map
    prediction_map_path = os.path.join("modeloutputs/AE_RF_prediction", fname)
    os.makedirs("modeloutputs/AE_RF_prediction", exist_ok=True)
    imsave(prediction_map_path, prediction_map)

print("Prediction maps saved in modeloutputs/AE_RF_prediction")