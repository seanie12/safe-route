import json

import numpy as np
import pandas as pd
from datasets import load_dataset

def get_dataset(dataset_name, valid_portion=0.1):
    # labels: response label for prompt-response, prmopt label for prompt
    if dataset_name == "harmbench":
        with open("data/harmbench/harmbench_eval.json") as f:
            harmbench_ds = json.load(f)

        prompts, prompts_label, responses, responses_label = [], [], [], []
        labels = []
        for x in harmbench_ds:
            prompts.append(x["prompt"])
            prompts_label.append(1)
            responses.append(x["response"])
            _label = 1 if x["label"] == "unsafe" else 0
            responses_label.append(_label)
            labels.append(_label)

    elif dataset_name == "wildguard-valid":
        wildguard_ds = load_dataset(
            "allenai/wildguardmix", "wildguardtrain", split="train"
        )
        df = pd.DataFrame(wildguard_ds)
        df = df[df["prompt_harm_label"].notna()]
        
        np.random.seed(42)
        indices = np.random.permutation(np.arange(len(df)))[
            : int(valid_portion * len(df))
        ]
        df_subset = df.iloc[indices]

        prompts, prompts_label, responses, responses_label = [], [], [], []
        labels = []
        for x in df_subset.itertuples():
            # prompt only
            prompts.append(x.prompt)
            prompt_label = int(x.prompt_harm_label == "harmful")
            prompts_label.append(prompt_label)
            
            labels.append(prompt_label)

            responses.append(None)
            responses_label.append(None)
            # prompt-response pair
            if x.response is not None and x.response_harm_label is not None:
                prompts.append(x.prompt)
                prompts_label.append(prompt_label)
                
                responses.append(x.response)
                response_label = int(x.response_harm_label == "harmful")
                responses_label.append(response_label)
                
                labels.append(response_label)

        assert len(prompts) == len(labels) == len(responses)
    
    elif dataset_name == "wildguard-test":
        wildguard_ds = load_dataset(
            "allenai/wildguardmix", "wildguardtest", split="test"
        )

        prompts, prompts_label, responses, responses_label = [], [], [], []
        labels = []
        for x in wildguard_ds:
            if (
                x["response_harm_label"] is not None
                and x["prompt_harm_label"] is not None
            ):
                prompts.append(x["prompt"])
                prompts_label.append(int(x["prompt_harm_label"] == "harmful"))
                responses.append(x["response"])
                responses_label.append(int(x["response_harm_label"] == "harmful"))
                labels.append(int(x["response_harm_label"] == "harmful"))

    elif dataset_name == "wildguard-test-prompt":
        wildguard_ds = load_dataset(
            "allenai/wildguardmix", "wildguardtest", split="test"
        )

        prompts, prompts_label, responses, responses_label = [], [], [], []
        labels = []
        responses = None
        responses_label = None
        for x in wildguard_ds:
            if x["prompt_harm_label"] is not None:
                prompts.append(x["prompt"])
                prompts_label.append(int(x["prompt_harm_label"] == "harmful"))
                labels.append(int(x["prompt_harm_label"] == "harmful"))

    elif dataset_name == "xstest":
        xstest_ds = load_dataset(
            "allenai/xstest-response", split="response_harmfulness"
        )
        prompts, prompts_label, responses, responses_label = [], [], [], []
        labels = []
        for x in xstest_ds:
            prompts.append(x["prompt"])
            prompts_label.append(int(x["prompt_type"] == "prompt_harmful"))
            responses.append(x["response"])
            responses_label.append(int(x["label"] == "harmful"))
            labels.append(int(x["label"] == "harmful"))

    elif dataset_name == "openai":
        with open("data/openai_moderation_eval.json", "r") as file:
            data = json.load(file)

        # Convert data to dataset format
        prompts = [x["prompt"] for x in data]
        prompts_label = [x["toxicity"] for x in data]
        responses = None
        responses_label = None
        labels = [x["toxicity"] for x in data]
    
    elif dataset_name == "toxic-chat":
        toxic_chat_ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="test")
        prompts, prompts_label, responses, responses_label = [], [], [], []
        labels = []
        responses = None
        responses_label = None
        for data in toxic_chat_ds:
            prompts.append(data["user_input"])
            prompts_label.append(data["toxicity"])
            labels.append(data["toxicity"])
    
    elif dataset_name == "simple":
        ds = load_dataset("Bertievidgen/SimpleSafetyTests", split="test")
        prompts, prompts_label, responses, responses_label = [], [], [], []
        labels = []
        responses = None
        responses_label = None
        for data in ds:
            prompts.append(data["prompt"])
            labels.append(1)
            prompts_label.append(1)
    
    else:
        raise NotImplementedError(f"No such datast {dataset_name}")

    return {
        "prompts": prompts,
        "responses": responses,
        "prompts_label": prompts_label,
        "responses_label": responses_label,
        "labels": labels
    }
