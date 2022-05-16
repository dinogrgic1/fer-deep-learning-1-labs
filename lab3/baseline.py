import torch
from torch import nn
import json
import numpy as np
from datetime import datetime
from sklearn.metrics import confusion_matrix

from utils import save_file_json, save_file_torch


SAVE_MODEL_DIR = 'saved'

class BaselineModel(nn.Module):
    def __init__(self, embedding_matrix, optimizer, lr=1e-4):
        super().__init__()
        self.fc1 = torch.nn.Linear(300, 150)
        self.fc2 = torch.nn.Linear(150, 150)
        self.fc3 = torch.nn.Linear(150, 1)
        self.loss =  torch.nn.BCEWithLogitsLoss()

        self.embedding_matrix = embedding_matrix
        self.optimizer = optimizer(self.get_parameters(), lr=lr)

        self.reset_parameters()

    def get_parameters(self):
        return [*self.fc1.parameters(), *self.fc2.parameters(), *self.fc3.parameters(), *self.embedding_matrix.parameters()]

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.embedding_matrix(x)
        y = torch.mean(y, dim=0 if len(y.shape) == 2 else 1)

        y = self.fc1(y)
        y = torch.relu(y)

        y = self.fc2(y)
        y = torch.relu(y)
        return self.fc3(y)

    def infer(self, x) -> int:
        with torch.no_grad():
            y = torch.sigmoid(self.forward(x))
            y = y.round().int().squeeze(-1)
        return y

    def fit(self, train, valid, epochs):
        loss_train = []
        loss_validation = []

        self.train()
        num_examples = len(train.dataset)
        
        for epoch in range(1, epochs + 1):
            loss_avg = []
            self.train()
            
            for i, batch in enumerate(train):
                loss = self.loss(self.forward(batch[0]).squeeze(-1), batch[1].float())
                loss.backward()
                
                loss_avg.append(float(loss))
                #torch.nn.utils.clip_grad_norm_(self.parameters(), args.clip)
                self.optimizer.step()

                if i % 100 == 0:
                    print(f'Training epoch #{epoch}, step {i * len(batch[0])}/{num_examples}, batch loss = {np.mean(loss_avg):.04f}')
                self.optimizer.zero_grad()

            with torch.no_grad():
                validation_metrics = self.evaluate(valid)

                print(f"Validation epoch #{epoch}\n"
                      f"\tloss: {validation_metrics['loss']:.04f}\n"
                      f"\tacc: {validation_metrics['acc']:.04f}\n"
                      f"\tf1: {validation_metrics['f1']:.04f}\n")

        
        now_str = datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
        save_file_torch(f'{SAVE_MODEL_DIR}/2', f'baseline_model_{now_str}.pth', self.state_dict())

        loss_validation = [str(x) for x in loss_validation]
        loss_train = [str(x) for x in loss_train]
        file = {"loss_validation": loss_validation, "loss_train": loss_train}
        save_file_json(f'{SAVE_MODEL_DIR}/2', f'model_{now_str}.json', file)

    def evaluate(self, validate):    
        x, y = zip(*[(entry[0].squeeze(0), entry[1]) for entry in list(validate)])

        self.eval()

        y_pred = []
        losses = []

        for _x, _y in zip(x, y):
            y_pred.append(self.infer(_x))
            y_pred_tensor = torch.tensor(y_pred[-1]).float().view(_y.shape)

            losses.append(float(self.loss(_y.float(), y_pred_tensor)))

        cm = confusion_matrix(np.array(y, dtype=np.int32), np.array(y_pred, dtype=np.int32))
        cm_diag = np.diag(cm)

        sums = [np.sum(cm, axis=y) for y in [None, 0, 1]]

        sums[0] = np.maximum(1, sums[0])
        for i in range(1, len(sums)):
            sums[i][sums[i] == 0] = 1

        accuracy = np.sum(cm_diag) / sums[0]
        precision, recall = [np.mean(cm_diag / x) for x in sums[1:]]
        f1 = (2 * precision * recall) / (precision + recall)

        return {"loss": float(np.mean(losses)),
                "acc": accuracy,
                "pr": precision,
                "re": recall,
                "f1": f1,
                "cm": np.ndarray.tolist(cm)}
