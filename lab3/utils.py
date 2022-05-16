import os
import torch
import json

def save_file_json(path, file, json_data):
    exists = os.path.exists(path)
    if not exists:
        os.makedirs(path)

    with open(f'{path}/{file}', 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file)

def save_file_torch(path, file, state_dict):
    exists = os.path.exists(path)
    if not exists:
        os.makedirs(path)

    torch.save(state_dict, f'{path}/{file}')
