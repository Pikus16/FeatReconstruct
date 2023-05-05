# FeatReconstruct
Reconstructing Features From Logits
 
Experiments
1. Given the predicted logits from a ResNet18, pretrained on ImageNet, can we learn the features with a single linear layer?
    * Training on CIFAR-100 train, and testing on CIFAR-100 test, we get a reconstruction, MSE loss of 0.28 total over the 10k test examples.
    * If we take these reconstructed features and then use the model's linear head, the top-1 prediction matches the original prediction 99.18% of the time.
2. Given the predicted logits from a ResNet18, pretrained on ImageNet, can we learn the features from a pretrained ResNet50 with a single linear layer?
    * Training on CIFAR-100 train, and testing on CIFAR-100 test, we get a reconstruction, MSE loss of 5.17 total over the 10k test examples.
    * If we take these reconstructed features and then use the model's linear head, the top-1 prediction matches the original prediction 99.47% of the time.