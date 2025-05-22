import json
import math
import os

import fire
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score)
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from dataset import get_dataset
from models import (CalibrationModel, Guardian, LlamaToxicClassifier, Oracle,
                    RandomModel, SafetyGuard)

DS = ["wildguard-test-prompt", "toxic-chat", "openai", "wildguard-test", "xstest", "harmbench", ]


def predict_ours(
    batch_size=8,
    version=3,
):
    device = torch.cuda.current_device()

    small_model = LlamaToxicClassifier(device, version="1b")
    if version in [1,2,3]:
        large_model = LlamaToxicClassifier(device, version=version)
    else:
        large_model = Guardian(device)
    # prompt_ckpt = f"save/guardian/bnn_guardian_{seed}/model.pt"
    
    ckpt = f"save/{version}/bnn_small/model.pt"
    model = SafetyGuard(
        ckpt,
        small_model,
        large_model,
    )

    model.eval()

    for dataset_name in DS:
        def predict_ours_inner(dataset_name, version):
            dataset = get_dataset(dataset_name)
            prompts = dataset["prompts"]
            responses = dataset["responses"]
            labels = dataset["labels"]

            ds = TensorDataset(torch.arange(len(prompts)))
            dataloader = DataLoader(ds, batch_size, shuffle=False)

            preds = []
            probs = []
            final_labels = []
            num_large = 0
            for batch in tqdm(dataloader, leave=False):
                ids = batch[0].tolist()
                batch_prompts = []
                batch_responses = []
                batch_labels = []

                for idx in ids:
                    batch_prompts.append(prompts[idx])
                    batch_labels.append(labels[idx])
                    if responses is not None:
                        batch_responses.append(responses[idx])
                    else:
                        batch_responses = None
                result = model(
                    batch_prompts,
                    batch_responses,
                    batch_labels,
                )

                num_large += result["num_large"]
                probs.append(result["probs"])
                preds.append(result["preds"])
                final_labels.append(result["final_labels"])


            probs = torch.cat(probs).numpy()
            preds = torch.cat(preds).numpy()
            final_labels = torch.cat(final_labels).numpy()

            acc = np.mean(final_labels == preds)
            f1 = f1_score(final_labels, preds)
            precision = precision_score(final_labels, preds)
            recall = recall_score(final_labels, preds)
            auc = average_precision_score(final_labels, probs)

            output_dir = os.path.join(
                "results", f"{dataset_name}", f"{version}")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            metric = {
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "accuracy": acc,
                "auc": auc,
                "large_ratio": num_large / len(prompts),
            }
            output_file = os.path.join(output_dir, f"ours.json")
            with open(output_file, "w") as f:
                json.dump(metric, f, indent=2)
            print(dataset_name)
            for k, v in metric.items():
                print(f"{k}: {v: .4f}")

        predict_ours_inner(dataset_name, version)

    del small_model, large_model
    torch.cuda.empty_cache()



def predict_oracle(batch_size=8, version=3):
    device = torch.cuda.current_device()
    small_model = LlamaToxicClassifier(device, version="1b")
    if version in [1,2,3]:
        large_model = LlamaToxicClassifier(device, version=version)
    else:
        large_model = Guardian(device)
    model = Oracle(small_model, large_model)

    for dataset_name in DS:

        def predict_oracle_inner(dataset_name, version):
            dataset = get_dataset(dataset_name)
            prompts = dataset["prompts"]
            responses = dataset["responses"]
            labels = dataset["labels"]

            ds = TensorDataset(torch.arange(len(prompts)))
            dataloader = DataLoader(ds, batch_size, shuffle=False)

            preds = []
            probs = []
            final_labels = []
            num_large = 0
            
            for batch in tqdm(dataloader, leave=False):
                ids = batch[0].tolist()
                batch_prompts = []
                batch_responses = []
                batch_labels = []

                for idx in ids:
                    batch_prompts.append(prompts[idx])
                    batch_labels.append(labels[idx])
                    if responses is not None:
                        batch_responses.append(responses[idx])
                    else:
                        batch_responses = None

                result = model(
                    batch_prompts,
                    batch_responses,
                    batch_labels,
                )

                num_large += result["num_large"]
                probs.append(result["probs"])
                preds.append(result["preds"])
                final_labels.append(result["final_labels"])

            probs = torch.cat(probs).numpy()
            preds = torch.cat(preds).numpy()
            final_labels = torch.cat(final_labels).numpy()

            acc = np.mean(final_labels == preds)
            f1 = f1_score(final_labels, preds)
            precision = precision_score(final_labels, preds)
            recall = recall_score(final_labels, preds)
            auc = average_precision_score(final_labels, probs)

            output_dir = os.path.join(
                "results", f"{dataset_name}", f"{version}")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            metric = {
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "accuracy": acc,
                "auc": auc,
                "large_ratio": num_large / len(prompts),
            }
            output_file = os.path.join(output_dir, "full_oracle.json")
            with open(output_file, "w") as f:
                json.dump(metric, f, indent=2)
            print(dataset_name)
            for k, v in metric.items():
                print(f"{k}: {v: .4f}")

        predict_oracle_inner(
            dataset_name=dataset_name, version=version
        )

    del small_model, large_model
    torch.cuda.empty_cache()


def predict_random(batch_size=8, version=3):
    device = torch.cuda.current_device()
    small_model = LlamaToxicClassifier(device, version="1b")
    if version in [1,2,3]:
        large_model = LlamaToxicClassifier(device, version=version)
    else:
        large_model = Guardian(device)
    model = RandomModel(small_model, large_model)

    for dataset_name in DS:

        def predict_random_inner(dataset_name, version):
            dataset = get_dataset(dataset_name)
            prompts = dataset["prompts"]
            responses = dataset["responses"]
            labels = dataset["labels"]

            ds = TensorDataset(torch.arange(len(prompts)))
            dataloader = DataLoader(ds, batch_size, shuffle=False)

            preds = []
            probs = []
            final_labels = []
            num_large = 0
            for batch in tqdm(dataloader, leave=False):
                ids = batch[0].tolist()
                batch_prompts = []
                batch_responses = []
                batch_labels = []

                for idx in ids:
                    batch_prompts.append(prompts[idx])
                    batch_labels.append(labels[idx])
                    if responses is not None:
                        batch_responses.append(responses[idx])
                    else:
                        batch_responses = None

                result = model(
                    batch_prompts,
                    batch_responses,
                    batch_labels,
                )

                num_large += result["num_large"]
                probs.append(result["probs"])
                preds.append(result["preds"])
                final_labels.append(result["final_labels"])

            probs = torch.cat(probs).numpy()
            preds = torch.cat(preds).numpy()
            final_labels = torch.cat(final_labels).numpy()

            acc = np.mean(final_labels == preds)
            f1 = f1_score(final_labels, preds)
            precision = precision_score(final_labels, preds)
            recall = recall_score(final_labels, preds)
            auc = average_precision_score(final_labels, probs)

            output_dir = os.path.join(
                "results", f"{dataset_name}", f"{version}")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            metric = {
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "accuracy": acc,
                "auc": auc,
                "large_ratio": num_large / len(prompts),
            }
            output_file = os.path.join(output_dir, "full_random.json")
            with open(output_file, "w") as f:
                json.dump(metric, f, indent=2)
            print(dataset_name)
            for k, v in metric.items():
                print(f"{k}: {v: .4f}")

        predict_random_inner(
            dataset_name=dataset_name, version=version)

    del small_model, large_model
    torch.cuda.empty_cache()


def get_batch_prob(model, data, batch_size):
    prompts = data["prompts"]
    responses = data["responses"]

    offset = 0
    num_batches = math.ceil(len(prompts) / batch_size)
    batch_response = None
    total_prob = 0.0
    total_count = 0
    
    for _ in tqdm(range(num_batches), leave=False, dynamic_ncols=True):
        batch_prompt = prompts[offset: offset + batch_size]
        if responses is not None:
            batch_response = responses[offset: offset+batch_size]
        probs = model.compute(batch_prompt, batch_response).exp().cpu().to(torch.float32)
        
        total_prob += probs.sum()
        total_count += probs.size(0)

        offset += len(batch_prompt)
    mean_prob = total_prob / total_count
    batch_prob = torch.tensor([1.0 - mean_prob, mean_prob])

    return batch_prob


def get_ts(model, data, batch_size):
    prompts = data["prompts"]
    responses = data["responses"]
    labels = data["labels"]
    device = model.device

    indices = torch.arange(len(prompts))
    ds = TensorDataset(indices)
    dataloader = DataLoader(ds, batch_size, shuffle=False)

    all_logits = []
    for batch in tqdm(dataloader, dynamic_ncols=True, leave=False):
        ids = batch[0]
        batch_prompts = [prompts[i.item()] for i in ids]
        if responses is not None:
            batch_responses = [responses[i.item()] for i in ids]
        with torch.no_grad():
            output = model.forward(batch_prompts, batch_responses)
            logits = output["logits"].to(torch.float32)

        all_logits.append(logits)

    all_logits = torch.cat(all_logits, dim=0)
    labels = torch.tensor(labels, dtype=torch.long).to(device)

    ts = torch.tensor(1.5).to(device).requires_grad_(True)
    opt = torch.optim.Adam([ts], lr=1e-2)
    for _ in tqdm(range(1000), leave=False, dynamic_ncols=True):
        opt.zero_grad()
        loss = F.cross_entropy(all_logits / ts, labels)
        loss.backward()
        opt.step()

    return ts.item()


def predict_bc(batch_size=8,  version=3):
    device = torch.cuda.current_device()
    small_model = LlamaToxicClassifier(device, version="1b")
    if version in [1,2,3]:
        large_model = LlamaToxicClassifier(device, version=version)
    else:
        large_model = Guardian(device)
    model = CalibrationModel(small_model, large_model)

    dataset = get_dataset("wildguard-valid", 1.0)
    small_batch_prob = get_batch_prob(small_model, dataset, batch_size)

    for dataset_name in DS:

        def predict_bc_inner(dataset_name, version):
            dataset = get_dataset(dataset_name)
            prompts = dataset["prompts"]
            responses = dataset["responses"]
            labels = dataset["labels"]

            ds = TensorDataset(torch.arange(len(prompts)))
            dataloader = DataLoader(ds, batch_size, shuffle=False)

            preds = []
            probs = []
            final_labels = []
            num_large = 0
            for batch in tqdm(dataloader, leave=False):
                ids = batch[0].tolist()
                batch_prompts = []
                batch_responses = []
                batch_labels = []

                for idx in ids:
                    batch_prompts.append(prompts[idx])
                    batch_labels.append(labels[idx])
                    if responses is not None:
                        batch_responses.append(responses[idx])
                    else:
                        batch_responses = None

                result = model(
                    batch_prompts,
                    batch_responses,
                    batch_labels,
                    batch_prob=small_batch_prob,
                )

                num_large += result["num_large"]
                probs.append(result["probs"])
                preds.append(result["preds"])
                final_labels.append(result["final_labels"])

            probs = torch.cat(probs).numpy()
            preds = torch.cat(preds).numpy()
            final_labels = torch.cat(final_labels).numpy()

            acc = np.mean(final_labels == preds)
            f1 = f1_score(final_labels, preds)
            precision = precision_score(final_labels, preds)
            recall = recall_score(final_labels, preds)
            auc = average_precision_score(final_labels, probs)

            output_dir = os.path.join(
                "results", f"{dataset_name}", f"{version}")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            metric = {
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "accuracy": acc,
                "auc": auc,
                "large_ratio": num_large / len(prompts),
            }
            output_file = os.path.join(output_dir, f"bc.json")
            with open(output_file, "w") as f:
                json.dump(metric, f, indent=2)
            print(dataset_name)
            for k, v in metric.items():
                print(f"{k}: {v: .4f}")

        predict_bc_inner(
            dataset_name=dataset_name,
            version=version,
        )

    del small_model, large_model
    torch.cuda.empty_cache()


def predict_ts(batch_size=8, version=3):
    device = torch.cuda.current_device()
    small_model = LlamaToxicClassifier(device, version="1b")
    if version in [1,2,3]:
        large_model = LlamaToxicClassifier(device, version=version)
    else:
        large_model = Guardian(device)
    model = CalibrationModel(small_model, large_model)

    dataset = get_dataset("wildguard-valid", 1.0)
    small_ts = get_ts(small_model, dataset, batch_size)
    for dataset_name in DS:

        def predict_ts_inner(dataset_name, version):
            dataset = get_dataset(dataset_name)
            prompts = dataset["prompts"]
            responses = dataset["responses"]
            labels = dataset["labels"]

            ds = TensorDataset(torch.arange(len(prompts)))
            dataloader = DataLoader(ds, batch_size, shuffle=False)

            preds = []
            probs = []
            final_labels = []
            num_large = 0
            for batch in tqdm(dataloader, leave=False):
                ids = batch[0].tolist()
                batch_prompts = []
                batch_responses = []
                batch_labels = []

                for idx in ids:
                    batch_prompts.append(prompts[idx])
                    batch_labels.append(labels[idx])
                    if responses is not None:
                        batch_responses.append(responses[idx])
                    else:
                        batch_responses = None

                result = model(
                    batch_prompts,
                    batch_responses,
                    batch_labels,
                    ts=small_ts,
                )

                num_large += result["num_large"]
                probs.append(result["probs"])
                preds.append(result["preds"])
                final_labels.append(result["final_labels"])

            probs = torch.cat(probs).numpy()
            preds = torch.cat(preds).numpy()
            final_labels = torch.cat(final_labels).numpy()

            acc = np.mean(final_labels == preds)
            f1 = f1_score(final_labels, preds)
            precision = precision_score(final_labels, preds)
            recall = recall_score(final_labels, preds)
            auc = average_precision_score(final_labels, probs)

            output_dir = os.path.join(
                "results", f"{dataset_name}", f"{version}")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            metric = {
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "accuracy": acc,
                "auc": auc,
                "large_ratio": num_large / len(prompts),
            }
            output_file = os.path.join(output_dir, f"ts.json")
            with open(output_file, "w") as f:
                json.dump(metric, f, indent=2)
            print(dataset_name)
            for k, v in metric.items():
                print(f"{k}: {v: .4f}")

        predict_ts_inner(
            dataset_name=dataset_name,
            version=version,
        )

    del small_model, large_model
    torch.cuda.empty_cache()


def get_context_prob(model):
    batch_probs = model.compute([" "], None).exp().cpu().to(torch.float32)
    batch_prob = torch.stack([1.0 - batch_probs, batch_probs], dim=1)[0]
    return batch_prob


def predict_cc(batch_size=8,  version=3):
    device = torch.cuda.current_device()
    small_model = LlamaToxicClassifier(device, version="1b")
    if version in [1,2,3]:
        large_model = LlamaToxicClassifier(device, version=version)
    else:
        large_model = Guardian(device)
    model = CalibrationModel(small_model, large_model)

    small_batch_prob = get_context_prob(small_model)

    for dataset_name in DS:

        def predict_cc_inner(dataset_name, version):
            dataset = get_dataset(dataset_name)
            prompts = dataset["prompts"]
            responses = dataset["responses"]
            labels = dataset["labels"]

            ds = TensorDataset(torch.arange(len(prompts)))
            dataloader = DataLoader(ds, batch_size, shuffle=False)

            preds = []
            probs = []
            final_labels = []
            num_large = 0
            
            for batch in tqdm(dataloader, leave=False):
                ids = batch[0].tolist()
                batch_prompts = []
                batch_responses = []
                batch_labels = []

                for idx in ids:
                    batch_prompts.append(prompts[idx])
                    batch_labels.append(labels[idx])
                    if responses is not None:
                        batch_responses.append(responses[idx])
                    else:
                        batch_responses = None

                result = model(
                    batch_prompts,
                    batch_responses,
                    batch_labels,
                    batch_prob=small_batch_prob,
                )

                num_large += result["num_large"]
                probs.append(result["probs"])
                preds.append(result["preds"])
                final_labels.append(result["final_labels"])

            probs = torch.cat(probs).numpy()
            preds = torch.cat(preds).numpy()
            final_labels = torch.cat(final_labels).numpy()

            acc = np.mean(final_labels == preds)
            f1 = f1_score(final_labels, preds)
            precision = precision_score(final_labels, preds)
            recall = recall_score(final_labels, preds)
            auc = average_precision_score(final_labels, probs)

            output_dir = os.path.join(
                "results", f"{dataset_name}", f"{version}")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            metric = {
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "accuracy": acc,
                "auc": auc,
                "large_ratio": num_large / len(prompts),
            }
            output_file = os.path.join(output_dir, f"cc.json")
            with open(output_file, "w") as f:
                json.dump(metric, f, indent=2)
            print(dataset_name)
            for k, v in metric.items():
                print(f"{k}: {v: .4f}")

        predict_cc_inner(
            dataset_name=dataset_name,
            version=version,
        )

    del small_model, large_model
    torch.cuda.empty_cache()


def predict_single(size="large", batch_size=8, version=3):
    device = torch.cuda.current_device()

    if size == "large":
        if version in [1,2,3]:
            model = LlamaToxicClassifier(device, version=version)
        else:
            model = Guardian(device)
    else:
        model = LlamaToxicClassifier(device, version="1b")

    for dataset_name in DS:

        def predict_single_inner(dataset_name, version):
            dataset = get_dataset(dataset_name)
            prompts = dataset["prompts"]
            responses = dataset["responses"]
            labels = dataset["labels"]

            ds = TensorDataset(torch.arange(len(prompts)))
            dataloader = DataLoader(ds, batch_size, shuffle=False)

            preds = []
            probs = []
            final_labels = []
            num_large = 0
            for batch in tqdm(dataloader, leave=False):
                ids = batch[0].tolist()
                batch_prompts = []
                batch_responses = []

                for idx in ids:
                    batch_prompts.append(prompts[idx])
                    if responses is not None:
                        batch_responses.append(responses[idx])
                    else:
                        batch_responses = None

                score = model.compute(
                    batch_prompts, batch_responses).float()
                prob = score.exp().cpu()
                pred = (prob > 0.5).long()

                if size == "large":
                    num_large += len(ids)
                else:
                    num_large = 0
                probs.append(prob)
                preds.append(pred)
                final_labels.append(labels)


            probs = torch.cat(probs).numpy()
            preds = torch.cat(preds).numpy()
            final_labels = np.array(labels)

            acc = np.mean(final_labels == preds)
            f1 = f1_score(final_labels, preds)
            precision = precision_score(final_labels, preds)
            recall = recall_score(final_labels, preds)
            auc = average_precision_score(final_labels, probs)

            output_dir = os.path.join(
                "results", f"{dataset_name}", f"{version}")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            metric = {
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "accuracy": acc,
                "auc": auc,
                "large_ratio": num_large / len(prompts),
            }
            output_file = os.path.join(output_dir, f"{size}.json")
            with open(output_file, "w") as f:
                json.dump(metric, f, indent=2)
            print(dataset_name)
            for k, v in metric.items():
                print(f"{k}: {v: .4f}")

        predict_single_inner(
            dataset_name=dataset_name, version=version)

    del model
    torch.cuda.empty_cache()


def run(batch_size=16, version=3):
    # predict_ts(batch_size=batch_size, version=version)
    # predict_bc(batch_size=batch_size, version=version)

    # predict_random(batch_size=batch_size,  version=version)
    # predict_oracle(batch_size=batch_size,  version=version)
    
    predict_ours(batch_size=batch_size, version=version)
    # predict_single("large", batch_size, version)
    # predict_single("small", batch_size, version)

if __name__ == "__main__":
    fire.Fire(run)
