# ViTforImbalancedData
Fine-tuning vision transformer for highly imbalanced multilabel dataset.

Medical datasets can be highly imablanced, and this problem becomes more difficult to deal with in case of multilabel datasets.
Vision Transformers have showed great promise and are considered state-of-the-art for a variety of computer vision tasks. But simply
fine-tuning them on target domain might not be enough when it comes to imabalanced multilabel dataset. 

Here, one such chest X-Ray dataset (NIH chest X-Ray) has been used. With a limited number of samples (of around 5000), and some labels
with less than a 100 samples, ViTs fail to give satisfactory results. ViTs are good at learning local and global features. This fact is
exploited and the ViT is fine-tuned on a multiclass chest X-Ray dataset (over 20000 samples). This helps the ViT to learn features
expected in a chest X-Ray. This fine-tuned ViT is then used as a feature extractor for the target dataset. The extracted features are
then fed to a linear classifier.
