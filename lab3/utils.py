import os
import torch
import json
import numpy as np
from sklearn.metrics import confusion_matrix


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

def model_metrics(y_real, y_pred):
    cm = confusion_matrix(np.array(y_real, dtype=np.int32), np.array(y_pred, dtype=np.int32))
    cm_diag = np.diag(cm)

    sums = [np.sum(cm, axis=y) for y in [None, 0, 1]]

    sums[0] = np.maximum(1, sums[0])
    for i in range(1, len(sums)):
        sums[i][sums[i] == 0] = 1

    accuracy = np.sum(cm_diag) / sums[0]
    precision, recall = [np.mean(cm_diag / x) for x in sums[1:]]
    f1 = (2 * precision * recall) / (precision + recall)

    return {"accuracy": accuracy,
            "recall": recall,
            "precision": precision,
            "f1": f1}