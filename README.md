# FeatReconstruct
Reconstructing Features From Logits
 
Experiments
1. Given the predicted logits from a ResNet18, pretrained on ImageNet, can we learn the features with a single linear layer?
    * CIFAR10
        * Training on CIFAR-10 train, and testing on CIFAR-10 test, we get a reconstruction, MSE loss of 0.28 total over the 10k test examples.
        * If we take these reconstructed features and then use the model's linear head, the top-1 prediction matches the original prediction 99.18% of the time.
    * CIFAR100
        * Training on CIFAR-100 train, and testing on CIFAR-100 test, we get a reconstruction, MSE loss of 0.224 total over the 10k test examples.
        * If we take these reconstructed features and then use the model's linear head, the top-1 prediction matches the original prediction 99.24% of the time.
2. Given the predicted logits from a ResNet18, pretrained on ImageNet, can we learn the features from a pretrained ResNet50 with a single linear layer?
    * CIFAR10
        * Training on CIFAR-10 train, and testing on CIFAR-10 test, we get a reconstruction, MSE loss of 5.17 total over the 10k test examples.
        * If we take these reconstructed features and then use the model's linear head, the top-1 prediction matches the original prediction 99.47% of the time.
    * CIFAR10
        * Training on CIFAR-100 train, and testing on CIFAR-100 test, we get a reconstruction, MSE loss of 6.52 total over the 10k test examples.
        * If we take these reconstructed features and then use the model's linear head, the top-1 prediction matches the original prediction 98.22% of the time.
3. Given the predicted logits from a ResNet18, pretrained on ImageNet, can we learn the features with a single linear layer, while training / testing on a different dataset?
    * CIFAR10 & Places365 val (Resnet18)
        * If we take these reconstructed features and then use the model's linear head, the top-1 prediction matches the original prediction 96.48% of the time.
    * CIFAR10 & Places365 val (Resnet18)
        * If we take these reconstructed features and then use the model's linear head, the top-1 prediction matches the original prediction 97.65 of the time.
    * CIFAR100 & Places365 val (Resnet50)
        * If we take these reconstructed features and then use the model's linear head, the top-1 prediction matches the original prediction 90.54% of the time.
4. Given the predicted logits from a ResNet18, pretrained on ImageNet, can we learn the features from a ViT CLIP model with a single linear layer?
    * Training on CIFAR-100 train, and testing on CIFAR-100 val
    * If we take these reconstructed features and then use the model's linear head, the top-1 prediction matches the original prediction  of the time.
    