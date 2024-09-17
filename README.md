# ViTforImbalancedData
This project was developed under partial fulfilment of MSc. AI U& ML course at the University of Birmimgham

# Background
Fine-tuning vision transformer as feature extractor for highly imbalanced multilabel dataset.

Medical datasets can be highly imbalanced. THere might not be enough data for rare diseases or emerging infectious diseases. The lack of enough positive samples makes it hard for models to learn distinguishing features. The problem becomes more severe when dealing with multilabel datasets. Moreover, most of the pre-trained models are trained on generic datasets that lack features seen in medical images like X-Rays, MRIs, CT Scans, etc.

Vision Transformers are considered state-of-the-art for a variety of computer vision tasks because of their ability to learn both local and global features by using attention mechanism on images. But, even ViT can fail to perform well on multilable classification task for a highly imbalanced dataset. To make this possible, ViT is first fine-tuned on a CXR dataset (Harvard Dataverse). This fine-tuned ViT is then used as a feature extractor and a FC linear classifier is trained for classification. 

A random sample of NIH Chest X-Ray dataset is used here. It has a small number of samples (of around 5000), and 15 labels for multilabel classification. There are less than a 100 samples for some labels. Techniques like data augmentation for minority classes and undersampling for majority classes are used. Samples are first fed to the fine-tuned ViT to get the features, which makes the input for the linear classifier. 

(Some experimentation is done prior to using the above solution to study the effects of small sample size, augmentations, using CNN based models vs transformer based models)

# Dataset

The dataset used is a randomly-sampled dataset of around 5000 images from the NIH CXR dataset. The sampled version can be found on Kaggle and is available to download here:

https://www.kaggle.com/datasets/nih-chest-xrays/sample/data

[dataset_analysis.ipynb](https://github.com/mriya98/ViTforImbalancedData/blob/main/dataset_analysis.ipynb) loads and analyses data, creates a metadata file with one-hot encoded labels and applies transformation to images.

# Experiments

MobileNet and ViT have been used for multi-label classification by fine-tuning on NIH dataset. The code can be found in "./experiments/" folder. However, neither of them show promising results with model accuracy and f1-score of around 0.45 - 0.55. Due to poor learing observed here, another architecture is proposed to enhance learning.

Class balancing techniques were also NIH dataset before fine-tuning the models discussed above. The parameters used and various other considerations are detailed in the thesis with supporting evidence from evaluation metrics.

# Proposed Architecture

ViT is fine-tuned on Harvard Dataverse dataset for Chest X-Rays. This is a multiclass classification dataset and the model is fine-tuned on this task. There are around 20000 images and 5 classes. The ViT learns local and global features in X-Rays well. This fine-tuned ViT is then used as feature extractor and linear classifier is later used for multilabel classification. The code can be found in [proposed_architecture.ipynb](https://github.com/mriya98/ViTforImbalancedData/blob/main/proposed_architecture.ipynb)

# Model Inference

The fine-tuned ViT and the trained classifier models can be used for feature extraction and classification respectively. Model weights of the linear classifier are available [here](https://drive.google.com/file/d/1-0fKYGXIJ6_UdO0vF36tqo9JljgORl2O/view?usp=sharing).

# Resources

The T4 GPU on Google Colab was used to train the models. Feature extraction, data augmentation and transformation to create input tensors for ViT were created using the High RAM mode on Colab for faster processing (can take upto 40 mins which is a waste of GPU runtime). These tensors are then saved to ease experimentation with the model and ensure that GPU runtime is not wasted on non-GPU intensive tasks.

# References

1. Basu, A., et al. (2021). Chest X-Ray Dataset for Respiratory Disease Classification, Harvard Dataverse.
2. Dosovitskiy, A., et al. (2020). "An image is worth 16x16 words: Transformers for image recognition at scale." arXiv preprint arXiv:2010.11929.

