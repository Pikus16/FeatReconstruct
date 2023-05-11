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
4. Can we learn the features of one model from the logits of another model?
    * On CIFAR100 train set, learn a linear layer to reconstruct the features of one model from the logits of another
    * On CIFAR100 test set, see how many of the top1 accuracies agree between taking the original features + model linear head, and the reconsturct features + same linear head
    ![CIFAR100 Cross Model](images/cifar100_crossmodel.png)
    * For reconstructing its own features from logits (diagonal)
        * Densenet, inception, and resnet are able to do it very well
        * ConvNext does it, but poorly
        * ViT and MLPMixer can't
    * Generally features can somewhat be reconstructed, but matching top1 is still poor
    * It appears that the difficulty comes on the logits side
        * ex: the ViT logits can't reconstruct anything. However, their features can be somewhat reconstructed from other models

5. Given the predicted logits from a ResNet18, pretrained on ImageNet, can we learn the features from a ViT CLIP model with a single linear layer?
    * Training on CIFAR-100 train, and testing on CIFAR-100 val
    * If we take these reconstructed features and then use the model's linear head, the top-1 prediction matches the original prediction  of the time.
    