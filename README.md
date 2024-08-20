# ViTforImbalancedData
Fine-tuning vision transformer for highly imbalanced multilabel dataset.

Medical datasets can be highly imablanced, and this problem becomes more difficult to deal with in case of multilabel datasets.
Vision Transformers have showed great promise and are considered state-of-the-art for a variety of computer vision tasks. But
fine-tuning them on target domain might not be enough when it comes to imabalanced multilabel dataset. 

Here, one such chest X-Ray dataset (NIH chest X-Ray) has been used. With a limited number of samples (of around 5000), and some labels
with less than a 100 samples, ViTs fail to give satisfactory results. ViTs are good at learning local and global features. This fact is
exploited and the ViT is fine-tuned on a multiclass chest X-Ray dataset (over 20000 samples). This helps the ViT to learn features
expected in a chest X-Ray. This fine-tuned ViT is then used as a feature extractor for the target dataset. The extracted features are
then fed to a linear classifier.

## Dataset

The dataset used is a randomly-sampled dataset of around 5000 images from the NIH CXR dataset. The sampled version can be found on
Kaggle and is available to download here:

https://www.kaggle.com/datasets/nih-chest-xrays/sample/data

dataset_analysis.ipynb loads and anallyses data, creates a metadata file with one-hot encoded labels

## Experiments

Mobilenet and ViT have been used for multi-label classification by fine-tuning on NIH dataset. The code can be found in 
"./experiments/" folder. However, neither of them show promising results with model accuracy and f1 coming to be around
0.45 - 0.55. Due to poor learing seen here, another technique is used to enhance learning.

## Proposed Architecture

ViT is fine-tuned on Harvard Dataverse dataset for Chest X-Rays. This is a multiclass classification dataset and the model is
fine-tuned on this task. There are around 20000 images and the ViT learns local and global features found in X-Rays well. This
fine-tuned ViT is then used as feature extractor and linear classifier is later used for multilabel classification.

The code can be found in proposed_architecture.ipynb

## Model Inference

Use evaluation.ipynb for inference

# SAVED MODELS

"models" folder has 4 models ready for inference (trained in proposed_architecuture). Can be loaded and experimented with.