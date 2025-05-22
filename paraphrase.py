import json
import os
import re

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from dataset import get_dataset
from tqdm import tqdm

import fire

def run(data_dir="data", gen_batch_size=512, num_rounds=7):
    dataset = get_dataset("wildguard-valid", 1.0)

    prompts, prompt_labels, responses, response_labels, labels = dataset["prompts"], dataset["prompts_label"], dataset["responses"], dataset["responses_label"], dataset["labels"]

    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    llm = LLM(model_id, dtype="bfloat16")
    sampling_params = SamplingParams(temperature=1.0, max_tokens=1024)    
    
    for round in range(num_rounds):
        print("-"*100)
        print("-"*40 + f"{round}-th round start" + "-"*40)
        print("-"*100)
        indices = torch.arange(len(prompts))
        ds = TensorDataset(indices)
        dataloader = DataLoader(ds, gen_batch_size, shuffle=False)
        
        gen_prompts = []
        gen_responses = []
        gen_prompt_labels = []
        gen_response_labels = []
        gen_labels = []
        for batch in tqdm(dataloader):
            ids = batch[0].tolist()
            
            batch_chats = []
            for i in ids:
                instruction = "Parapharse the following prompt and response respectively while preserving their original semantics. Adhere strictly to the following format. First do not include original prompt and response.  Second, start the paraphrased prompt with \"prompt:\". Third, start the paraphrased response with \"response:\". Fourth, if response is given as None, just provide it as None.\n"
                instruction += f"prompt: {prompts[i]}\n\nresponse: {responses[i]}\n\n"
                
                chat = tokenizer.apply_chat_template([{"role": "user", "content": instruction}], tokenize=False, add_generation_prompt=True)
                batch_chats.append(chat)
                
            outputs = llm.generate(batch_chats, sampling_params)
            for i, output in enumerate(outputs):
                j_i = ids[i]
                generated_text = output.outputs[0].text
                
                # Define regex pattern
                pattern = r"prompt:\s*(.*?)\s*response:\s*(.*)"
                match = re.search(pattern, generated_text, re.DOTALL)
                if match:
                    prompt = match.group(1).strip()
                    response = match.group(2).strip()
                    if response == "None":
                        response = None
                    
                    prompt_label = prompt_labels[j_i]
                    response_label = response_labels[j_i]
                    label = labels[j_i]
                    
                    gen_prompts.append(prompt)
                    gen_responses.append(response)
                    gen_prompt_labels.append(prompt_label)
                    gen_response_labels.append(response_label)
                    gen_labels.append(label)
                else:
                    continue    
        
        gen_dataset = {
            "prompts": gen_prompts,
            "responses": gen_responses,
            "prompts_label": gen_prompt_labels,
            "responses_label": gen_response_labels,
            "labels": gen_labels
        }
        
        os.makedirs(os.path.join(data_dir, "aug"), exist_ok=True)
        with open(os.path.join(data_dir, "aug", f"round{round}.json"), "w") as f:
            json.dump(gen_dataset, f)

if __name__ == "__main__":
    fire.Fire(run)