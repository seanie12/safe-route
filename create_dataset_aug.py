import json
import os
import gc

import fire
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models import LlamaToxicClassifier, Guardian

LAYER_INDICES = [-1]

@ torch.no_grad()
def get_features_labels_items(batch_size, small_model, large_model, prompts, responses, labels):
    labels = torch.LongTensor(labels)
    
    dataset = TensorDataset(torch.arange(len(prompts)))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    small_preds = []
    features = []
    small_model.eval()
    for batch in tqdm(dataloader, dynamic_ncols=True, leave=False, desc="small model"):
        ids = batch[0].tolist()
        
        batch_prompt = [prompts[i] for i in ids]
        if responses is not None:
            batch_response = [responses[i] for i in ids]
        else:
            batch_response = None
        with torch.no_grad():
            output = small_model.forward(batch_prompt, batch_response, layer_idx=-1)
        hidden = output["features"].cpu().float()
        features.append(hidden)
        
        score = output["unsafe_logprob"].exp().cpu().float()
        pred = (score > 0.5).long()
        small_preds.append(pred)
    
    features = torch.cat(features, dim=0)
    small_preds = torch.cat(small_preds, dim=0)

    large_preds = []
    large_model.eval()
    for batch in tqdm(dataloader, leave=False, dynamic_ncols=True, desc="large model"):
        ids = batch[0].tolist()
        batch_prompt = [prompts[i] for i in ids]
        if responses is not None:
            batch_response = [responses[i] for i in ids]
        else:
            batch_response = None
        score = large_model.compute(batch_prompt, batch_response).exp().float().cpu()        
        preds = (score > 0.5).long()
        large_preds.append(preds)
            
        del score, preds     
        gc.collect()
        torch.cuda.empty_cache()

    large_preds = torch.cat(large_preds, dim=0)
    
    large_mask = torch.logical_and(small_preds != labels, large_preds == labels)
    targets = large_mask.long()

    items = []
    for i in range(len(prompts)):
        items.append(
            {"prompt": prompts[i], "response": responses[i], 
            "label": targets[i].item(), "harmfulness": labels[i].item()})
    return features, targets, items
    

def run(round=0, data_dir="data", batch_size=16, version=3):
    with open(os.path.join(data_dir, "aug", f"round{round}.json"), "r") as f:
        dataset = json.load(f)
    
    prompts = dataset["prompts"]
    print(len(prompts)) 
    responses = dataset["responses"]
    labels = dataset["labels"]

    device = torch.cuda.current_device()

    small_model = LlamaToxicClassifier(device=device, version="1b")
    small_model.eval()

    if version == "guardian":
        large_model = Guardian(device=device)
    else:
        large_model = LlamaToxicClassifier(device=device, version=version)
    large_model.eval()
    
    features, targets, _ = get_features_labels_items(
        batch_size, small_model, large_model, prompts, responses, labels)
    
    output_dir = os.path.join(data_dir, f"{version}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # small features
    torch.save(features, os.path.join(output_dir, f"round{round}_features.pt"))
    
    # targets
    torch.save(targets, os.path.join(output_dir, f"round{round}_labels.pt"))
    

if __name__ == "__main__":
    fire.Fire(run)

        
        

