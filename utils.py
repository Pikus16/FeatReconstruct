from torchvision import transforms, models, datasets
import numpy as np
import os
from torchvision.transforms.functional import InterpolationMode
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy

def get_transforms(dataset_name):
    if dataset_name == 'cifar10' or dataset_name == 'cifar100' or dataset_name == "places365":
        mean=(0.485, 0.456, 0.406)
        crop_size = 224
        std=(0.229, 0.224, 0.225)
        return transforms.Compose([
            transforms.CenterCrop((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    elif dataset_name == 'imagenet':
        resize_size = 256
        crop_size = 224
        mean=(0.485, 0.456, 0.406)
        std=(0.229, 0.224, 0.225)

        return transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(crop_size),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=mean, std=std),
        ]
        )
    else:
        raise ValueError(f"unknown dataset name {dataset_name}")

def get_dataset(dataset_name, data_dir='/home/ubuntu/data',test_transform=None, get_train=True):
    dataset_name = dataset_name.lower()
    if test_transform is None:
        test_transform = get_transforms(dataset_name)
    
    if dataset_name == "cifar10":
        base_dir = os.path.join(data_dir, 'cifar10')
        test_set = datasets.CIFAR10(root=base_dir, train=False, download=True, transform=test_transform)
        if get_train:
            train_set = datasets.CIFAR10(root=base_dir, train=True, download=True, transform=test_transform)
    elif dataset_name == "cifar100":
        base_dir = os.path.join(data_dir, 'cifar100')
        test_set = datasets.CIFAR100(root=base_dir, train=False, download=True, transform=test_transform)
        if get_train:
            train_set = datasets.CIFAR100(root=base_dir, train=True, download=True, transform=test_transform)
    elif dataset_name == 'imagenet':
        base_dir = os.path.join(data_dir, 'Imagenet/ILSVRC/Data/CLS-LOC/')
        test_set = datasets.ImageFolder(f'{base_dir}/val/', transform=test_transform)
        if get_train:
            train_set = datasets.ImageFolder(f'{base_dir}/train/', transform=test_transform)
    elif dataset_name == "places365":
        base_dir = os.path.join(data_dir, 'Places365_small')
        test_set = datasets.Places365(root=base_dir, split='val',small=True, download=False, transform=test_transform)
        if get_train:
            train_set = datasets.Places365(root=base_dir, split='train-tandard',small=True, download=False, transform=test_transform)
    else:
        raise ValueError(f"unknown dataset name {dataset_name}")
    
    if get_train:
        return train_set, test_set
    else:
        return test_set

class ModelWithLogits(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.original_fc = copy.deepcopy(model.fc)
        self.feature_extraxtor = copy.deepcopy(model)
        self.feature_extraxtor.fc = torch.nn.Identity()

    def forward(self, x):
        feats = self.feature_extraxtor(x)
        logits = self.original_fc(feats)
        return feats, logits

def get_model(model_name):
    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=True)
    else:
        raise ValueError(f"unknown model name {model_name}")
    return ModelWithLogits(model)



def get_features(model, dataset, batch_size=32, num_workers=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert isinstance(model, ModelWithLogits)
    model.to(device)
    model.eval()
    dataloader = DataLoader(dataset,batch_size=batch_size, num_workers=num_workers,shuffle=False)
    all_feats, all_logits, all_labels = [], [], []
    for (images, label) in tqdm(dataloader, total=len(dataset) // batch_size):
        with torch.no_grad():
            output_feats, output_logits = model(images.to(device))
            all_feats.append(output_feats.cpu().numpy())
            all_logits.append(output_logits.cpu().numpy())
            all_labels.append(label.numpy())
    all_feats = np.concatenate(all_feats)
    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    return all_feats, all_logits, all_labels

class FeatDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels=None):
        self.features = torch.FloatTensor(features)
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = self.features[idx]
        if self.labels is None:
            return feat
        else:
            return feat, self.labels[idx]

def get_logits_from_feats(feats, lin, batch_size=32, num_workers=8):
    # Compute logits given features
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lin.to(device)
    lin.eval()
    all_preds = []
    dataloader = DataLoader(FeatDataset(feats), batch_size=batch_size, num_workers=num_workers,shuffle=False)
    for batch_feats in tqdm(dataloader, total=len(feats) // batch_size):
        with torch.no_grad():
            pred = lin(batch_feats.to(device)).cpu().numpy()
            all_preds.append(pred)
    all_preds = np.concatenate(all_preds)
    return all_preds

def get_recon_loss(recon_model, logits, features, batch_size=16):
    rec_loss = torch.nn.MSELoss()
    recon_model.eval()
    test_loss = 0
    for b in range(0, len(logits), batch_size):
        with torch.no_grad():
            x = logits[b:b+batch_size]
            y = features[b:b+batch_size]
            pred_y = recon_model(x)
            loss = rec_loss(pred_y, y)
            test_loss += loss.item()
    return test_loss

def learn_reconstruct(logits, features, test_logits, test_features):
    lin = torch.nn.Linear(logits.shape[1], features.shape[1])
    rec_loss = torch.nn.MSELoss()
    epochs = 100
    batch_size = 16
    optimizer = torch.optim.SGD(lin.parameters(), lr = 0.1, momentum = 0.9)
    
    logits = torch.FloatTensor(logits)
    features = torch.FloatTensor(features)
    test_logits = torch.FloatTensor(test_logits)
    test_features = torch.FloatTensor(test_features)

    for i in range(epochs):
        total_loss = 0
        for b in range(0, len(logits), batch_size):
            optimizer.zero_grad()
            x = logits[b:b+batch_size]
            y = features[b:b+batch_size]
            pred_y = lin(x)
            loss = rec_loss(pred_y, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if i % 10 == 0:
            test_loss = get_recon_loss(lin, test_logits, test_features, batch_size=batch_size)
            print(f"Epoch {i}, train loss={total_loss}, test loss = {test_loss}")
            lin.train()
            
    return lin

def do_feat_recon(logits, recon_model_):
    batch_size = 16
    logits = torch.FloatTensor(logits)
    all_feats = []
    with torch.no_grad():
        for b in range(0, len(logits), batch_size):
            x = logits[b:b+batch_size]
            pred_y = recon_model_(x).numpy()
            all_feats.append(pred_y)
    all_feats = np.concatenate(all_feats)
    return all_feats