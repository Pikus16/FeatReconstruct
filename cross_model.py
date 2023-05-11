import numpy as np
from utils import get_model, get_transforms, get_dataset, get_features, get_recon_loss, learn_reconstruct, do_feat_recon, get_logits_from_feats
import json

test_transform = get_transforms('cifar100')
train_set, test_set = get_dataset('cifar100', test_transform=test_transform, get_train=True)

TIMM_MODEL_NAMES = ['vit_base_patch16_224_in21k','densenet121', 'inception_v4', 'convnext_base', 'mixer_b16_224']
TORCHVISION_MODEL_NAMES = ['resnet18', 'resnet50']

all_models = {}
for model_name in TIMM_MODEL_NAMES:
    all_models[f'timm_{model_name}'] = get_model(model_name, model_src='timm')
for model_name in TORCHVISION_MODEL_NAMES:
    all_models[f'torchvision_{model_name}'] = get_model(model_name, model_src='torchvision')

all_train_feats, all_test_feats = {}, {}
all_train_logits, all_test_logits = {}, {}
for model_name, model in all_models.items():
    feats_train, logits_train, _ = get_features(model, train_set)
    feats_test, logits_test, _ = get_features(model, test_set)
    all_train_feats[model_name] = feats_train
    all_test_feats[model_name] = feats_test
    all_train_logits[model_name] = logits_train
    all_test_logits[model_name] = logits_test

avg_agree = {}
for model_a in all_models:
    avg_agree_a = {}
    for model_b in all_models:
        # go through every pair of models (A and B)
        # Learn to reconstruct the logits of A from the features of B
        recon_model = learn_reconstruct(
            all_train_logits[model_a], all_train_feats[model_b],
            all_test_logits[model_a], all_test_feats[model_b]
        )
        # Reconstruct the features of B from the logits of A on the test set
        recon_test_feats = do_feat_recon(all_test_logits[model_a], recon_model)
        # Predict the logits of B on the reconstructed features of B from A
        preds_recon = get_logits_from_feats(recon_test_feats, all_models[model_b].original_fc)
        # Compare the top1 predictions on model B of reconstructed vs not
        top_recon = np.argmax(preds_recon, axis=1)
        top_orig = np.argmax(all_test_logits[model_b], axis=1)
        avg_agree_a[model_b] = np.mean(top_orig == top_recon)
    avg_agree[model_a] = avg_agree_a
with open('agreement.json', 'w') as f:
    json.dump(avg_agree, f, indent=2)