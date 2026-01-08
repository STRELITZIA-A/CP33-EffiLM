import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json


def check_string(a, string_list):
    n = 0
    for s in string_list:
        if s in a:
            n += 1
    if n==0 :
        return False
    else:
        return True


def get_all_stop_tokens(tokenizer):
    """
    Extract all potential stop tokens from a tokenizer
    """
    stop_tokens = set()
    
    # Get eos token
    if tokenizer.eos_token_id is not None:
        stop_tokens.add(tokenizer.eos_token_id)
    
    return list(stop_tokens)

def merge_models_on_cpu(model_path1, model_path2, weight_ratio1, weight_ratio2, save_path):
    """
    Merge two language models and save in HuggingFace format
    """
    print("Loading first model on CPU...")
    model1 = AutoModelForCausalLM.from_pretrained(
        model_path1,
        device_map="cpu",
        torch_dtype="auto"
    )

    print(model1)


    state_dict1 = model1.state_dict()
    del model1
    torch.cuda.empty_cache()
    
    print("Loading second model on CPU...")
    model2 = AutoModelForCausalLM.from_pretrained(
        model_path2,
        device_map="cpu",
        torch_dtype="auto"
    )
    
    state_dict2 = model2.state_dict()
    del model2
    torch.cuda.empty_cache()
    
    print("Merging weights...")
    merged_state_dict = {}
    
    for key in state_dict1.keys():

        merged_state_dict[key] = (state_dict1[key]*weight_ratio1 + state_dict2[key]*weight_ratio2)

        if "lm_head" in key:
            merged_state_dict[key][151643,:] = state_dict1[key][151643,:]
            merged_state_dict[key][151645,:] = state_dict1[key][151645,:]

    # Load a new model instance to save merged weights
    print("Saving merged model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path1,
        device_map="cpu",
        torch_dtype="auto"
    )
    base_model.load_state_dict(merged_state_dict)
    
    # Save the merged model in HuggingFace format
    os.makedirs(save_path, exist_ok=True)
    base_model.save_pretrained(save_path)
    
    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path1)
    tokenizer.save_pretrained(save_path)


    print("Loading tokenizers and collecting stop tokens...")
    tokenizer1 = AutoTokenizer.from_pretrained(model_path1)
    tokenizer2 = AutoTokenizer.from_pretrained(model_path2)
    # Collect stop tokens from both models
    stop_tokens1 = get_all_stop_tokens(tokenizer1)
    stop_tokens2 = get_all_stop_tokens(tokenizer2)
    # Merge stop tokens
    merged_stop_tokens = list(set(stop_tokens1 + stop_tokens2))

    with open(os.path.join(save_path, "generation_config.json"), 'r', encoding='utf-8') as file:
        generation_config = json.load(file)
    generation_config['eos_token_id'] = merged_stop_tokens
    with open(os.path.join(save_path, "generation_config.json"), "w") as f:
        json.dump(generation_config, f)
    del tokenizer1, tokenizer2

    
    del base_model, merged_state_dict, state_dict1, state_dict2
    torch.cuda.empty_cache()
    
    print(f"Merged model saved to {save_path}")

# Example usage:
if __name__ == "__main__":

    path = [    
        "/mnt/data/MODEL/Qwen/Qwen2.5-Math-7B",
        "/mnt/data/MODEL/Qwen/Qwen2.5-Math-1.5B",
        "/mnt/data/MODEL/Qwen/Qwen2.5-Math-7B-Instruct",
        "/mnt/data/MODEL/Qwen/Qwen2.5-Math-1.5B-Instruct",
        "/mnt/data/MODEL/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "/mnt/data/MODEL/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    ]
    
    model_path1 = path[4]
    model_path2 = path[3]

    merged_model_path = "/mnt/data/Z_OUYANGXUANXIN/EXP/NUS-FYP/merged_models/linear/1.5B_inst_r1base"
    
    # Step 1: Merge models and save in HuggingFace format
    merge_models_on_cpu(
        model_path1=model_path1,
        model_path2=model_path2,
        weight_ratio1=0.8,
        weight_ratio2=0.2,
        save_path=merged_model_path
    )
    

