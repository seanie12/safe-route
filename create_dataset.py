import json
import os

import fire
import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models import LlamaToxicClassifier, Guardian
from dataset import get_dataset


def get_features_labels_items(batch_size, small_model, large_model, prompts, responses, labels, layer_idx=-1):
    labels = torch.LongTensor(labels)
    
    dataset = TensorDataset(torch.arange(len(prompts)))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    small_preds = []
    features = []
    for batch in tqdm(dataloader, dynamic_ncols=True, leave=False, desc="small model"):
        ids = batch[0].tolist()
        
        batch_prompt = [prompts[i] for i in ids]
        if responses is not None:
            batch_response = [responses[i] for i in ids]
        else:
            batch_response = None
        with torch.no_grad():
            output = small_model.forward(batch_prompt, batch_response, layer_idx=layer_idx)
        hidden = output["features"].cpu().float()
        features.append(hidden)
        
        score = output["unsafe_logprob"].exp().cpu().float()
        pred = (score > 0.5).long()
        small_preds.append(pred)
    
    features = torch.cat(features, dim=0)
    small_preds = torch.cat(small_preds, dim=0)

    large_preds = []
    large_features = []
    large_model.eval()
    for batch in tqdm(dataloader, leave=False, dynamic_ncols=True, desc="large model"):
        ids = batch[0].tolist()
        batch_prompt = [prompts[i] for i in ids]
        if responses is not None:
            batch_response = [responses[i] for i in ids]
        else:
            batch_response = None
        with torch.no_grad():
            output = large_model.forward(batch_prompt, batch_response, layer_idx=layer_idx)
        hidden = output["features"].float().cpu()
        score = output["unsafe_logprob"].exp().float().cpu()
        preds = (score > 0.5).long()
        
        large_preds.append(preds)
        large_features.append(hidden)

    large_preds = torch.cat(large_preds, dim=0)
    large_features = torch.cat(large_features)
    
    large_mask = torch.logical_and(small_preds != labels, large_preds == labels)
    targets = large_mask.long()

    items = []
    for i in range(len(prompts)):
        items.append(
            {"prompt": prompts[i], "response": responses[i], 
            "label": targets[i].item(), "harmfulness": labels[i].item()})
    return features, targets, items
    

def run(data_dir="data", batch_size=8, version=3, layer_idx=-1):
    dataset = get_dataset("wildguard-valid", 1.0)
    
    prompts = dataset["prompts"]
    print(len(prompts)) 
    responses = dataset["responses"]
    labels = dataset["labels"]

    device = torch.cuda.current_device()

    small_model = LlamaToxicClassifier(device=device, version="1b")
    small_model.eval()
    if version == "guardian":
        large_model = Guardian(device)
        print("guardian")
    elif version in [1,2,3]:
        print(f"llama-guard-{version}")
        large_model = LlamaToxicClassifier(device=device, version=version)
    else:
        raise NotImplementedError()
    large_model.eval()
    
    features, targets, items = get_features_labels_items(
        batch_size, small_model, large_model, prompts, responses, labels, layer_idx=layer_idx)


    X = np.arange(len(items))
    y = [item["label"] for item in items]
    y = np.array(y)
    # train / validation split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    train_idx, val_idx = next(sss.split(X, y))

    train_items = [items[idx] for idx in train_idx]
    val_items = [items[idx] for idx in val_idx]
    
    train_features = torch.stack([features[idx] for idx in train_idx], dim=0)
    train_targets = torch.FloatTensor([targets[idx].item() for idx in train_idx])
    
    val_features = torch.stack([features[idx] for idx in val_idx], dim=0)
    val_targets = torch.FloatTensor([targets[idx] for idx in val_idx])
    
    output_dir = os.path.join(data_dir, f"{version}", f"layer_{layer_idx}")    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(os.path.join(output_dir,  "1B_8B_train.json"), "w") as f:
        json.dump(train_items, f, indent=2)

    torch.save(train_features, os.path.join(output_dir, "train_features.pt"))
    torch.save(train_targets, os.path.join(output_dir, "train_labels.pt"))

    torch.save(val_features, os.path.join(output_dir, "val_features.pt"))
    torch.save(val_targets, os.path.join(output_dir, "val_labels.pt"))
    
    print(f"# 1: {torch.sum(targets).item()} / {targets.size(0)}")
    with open(os.path.join(output_dir, "1B_8B_val.json"), "w") as f:
        json.dump(val_items, f, indent=2)

if __name__ == "__main__":
    fire.Fire(run)

        
        

