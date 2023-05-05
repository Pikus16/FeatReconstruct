# FeatReconstruct
Reconstructing Features From Logits
 
Experiments
1. Given a resnet18, pretrained on ImageNet, learns the features from the logits with a single linear layer. 
    * Training on CIFAR-100 train, and testing on CIFAR-100 test, we get a reconstruction, MSE loss of 0.28 total over the 10k test examples.
    * If we take these reconstructed features and then use the model's linear head, the top-1 prediction matches the original prediction 99.18% of the time.