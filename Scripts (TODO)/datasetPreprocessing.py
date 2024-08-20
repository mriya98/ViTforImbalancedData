import datasets
from datasets import Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

class datasetPreprocessing:

    dataset_folder = './dataset/samples/'
    csv_file = './dataset/'
    '''
    Craete a metadata json file with image filenames and one-hot encoded labels
    '''
    def create_metadata(self, dataset_path, csv_path):

        # Load data
        self.dataset_folder = dataset_path
        self.csv_file = csv_path

        data = pd.read_csv(self.csv_file)

        # Converts the format of each label in the dataframe from "LabelA|LabelB|LabelC"
        # into ["LabelA", "LabelB", "LabelC"], concatenates the
        # lists together and removes duplicate labels
        unique_labels = np.unique(
            data["Finding_Labels"].str.split("|").aggregate(np.concatenate)
        ).tolist()

        #print(f"Dataset contains the following labels:\n{unique_labels}")

        # Transform labels to n hot encoded array
        label_index = {v: i for i, v in enumerate(unique_labels)}

        def string_to_N_hot(string: str):
            true_index = [label_index[cl] for cl in string.split("|")]
            label = np.zeros((len(unique_labels),), dtype=float)
            label[true_index] = 1
            return label

        data["labels"] = data["Finding_Labels"].apply(string_to_N_hot)