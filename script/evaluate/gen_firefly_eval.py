"""
生成人工评测数据集的结果
"""

from transformers import AutoTokenizer
from transformers.generation.utils import GenerationConfig
import torch
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append("../../")
from component.utils import ModelUtils


def load_model_and_tokenizer(model_name_or_path):
    # 加载模型
    model = ModelUtils.load_model(model_name_or_path).eval()
    if model_name_or_path == 'baichuan-inc/Baichuan-13B-Chat':
        model.generation_config = GenerationConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if model.config.model_type == 'llama' else True
    )
    # QWenTokenizer比较特殊，pad_token_id、bos_token_id、eos_token_id均为None。eod_id对应的token为<|endoftext|>
    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id
    return model, tokenizer


def generate(model, tokenizer, model_name_or_path, instruction, device='cuda'):
    instruction = instruction.strip()
    model_name = model_name_or_path.split('/')[-1]
    if 'firefly' in model_name:
        input_ids = tokenizer(instruction, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        bos_token_id = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).to(device)
        eos_token_id = torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long).to(device)
        input_ids = torch.concat([bos_token_id, input_ids, eos_token_id], dim=1)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids, max_new_tokens=2048, do_sample=True,
                top_p=0.9, temperature=0.35, repetition_penalty=1.0,
                eos_token_id=tokenizer.eos_token_id
            )
        outputs = outputs.tolist()[0][len(input_ids[0]):]
        response = tokenizer.decode(outputs)
        response = response.strip().replace(tokenizer.eos_token, "").strip()
    elif model_name_or_path == 'baichuan-inc/Baichuan-13B-Chat':
        messages = [{"role": "user", "content": instruction}]
        response = model.chat(tokenizer, messages)
    elif model_name_or_path == 'Linly-AI/Chinese-LLaMA-2-13B-hf':
        prompt = f"### Instruction:{instruction.strip()}  ### Response:"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")
        outputs = model.generate(input_ids, do_sample=True, max_new_tokens=2048, top_k=10, top_p=0.85,
                                temperature=1, repetition_penalty=1.15, eos_token_id=2, bos_token_id=1,
                                pad_token_id=0)
        outputs = outputs.tolist()[0][len(input_ids[0]):]
        response = tokenizer.decode(outputs)
        response = response.strip().replace(tokenizer.eos_token, "").strip()
    elif model_name_or_path == 'BELLE-2/BELLE-Llama2-13B-chat-0.4M':
        prompt = f"Human: \n{instruction.strip()} \n\nAssistant: \n"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        generate_ids = model.generate(input_ids, max_new_tokens=1024, do_sample=True, top_k=30, top_p=0.85,
                                        temperature=0.5, repetition_penalty=1.2, eos_token_id=2, bos_token_id=1,
                                        pad_token_id=0)
        output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        response = output[len(prompt):]
    return response


def main():
    # model_name_or_path = 'YeungNLP/firefly-baichuan-13b'
    # save_file = './firefly-baichuan-13b-chat.xlsx'
    eval_file = '../../data/firefly-eval.csv'

    model_name_or_path = 'YeungNLP/firefly-llama2-13b-chat'
    save_file = './firefly-llama2-13b-chat.xlsx'

    # 加载模型
    print(f'Loading model from {model_name_or_path}')
    model, tokenizer = load_model_and_tokenizer(model_name_or_path)
    # 读取测试集
    df = pd.read_excel(eval_file)
    print(len(df))

    result = []
    for _, row in tqdm(df.iterrows()):
        kind = row['kind']
        instruction = row['instruction'].strip()
        response = generate(model, tokenizer, model_name_or_path, instruction).strip()
        result.append({
            'kind': kind,
            'instruction': instruction,
            'output': response,
            'model': model_name_or_path
        })
    df = pd.DataFrame(result)
    df.to_excel(save_file, index=False)


if __name__ == '__main__':
    main()
