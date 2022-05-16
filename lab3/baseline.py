import torch
from torch import nn
import json
import numpy as np
from datetime import datetime
from sklearn.metrics import confusion_matrix

from utils import save_file_json, save_file_torch, model_metrics


SAVE_MODEL_DIR = 'saved'
SAVE_MODEL_NAME = 'baseline'
TASK = 'task_2'

class BaselineModel(nn.Module):
    def __init__(self, embedding_matrix, optimizer, lr=1e-4, *args, **kwargs):
        super().__init__()
        
        self.fc1 = torch.nn.Linear(300, 150)
        self.fc2 = torch.nn.Linear(150, 150)
        self.fc3 = torch.nn.Linear(150, 1)
        self.loss =  torch.nn.BCEWithLogitsLoss()
        
        self.kwargs = kwargs
        self.embedding_matrix = embedding_matrix
        self.lr = lr
        self.optimizer = optimizer(self.get_parameters(), lr=self.lr)

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

    def fit(self, train, valid, test, epochs):
        metrics_train = []
        metrics_validation = []

        self.train()
        num_examples = len(train.dataset)
        
        for epoch in range(1, epochs + 1):
            loss_avg = []
            self.train()
            y_pred = []
            y_real = []
            
            for i, batch in enumerate(train):
                loss = self.loss(self.forward(batch[0]).squeeze(-1), batch[1].float())
                loss.backward()
                
                y_pred.append(self.infer(batch[0]))
                y_real.append(batch[1])
                loss_avg.append(float(loss))
                self.optimizer.step()

                if i % 100 == 0:
                    print(f'Training epoch #{epoch}, step {i * len(batch[0])}/{num_examples}, batch loss = {np.mean(loss_avg):.04f}')
                self.optimizer.zero_grad()

            y_pred = torch.cat(y_pred)
            y_real = torch.cat(y_real)

            train_metrics = model_metrics(y_real, y_pred)
            train_metrics['loss'] = np.average(loss_avg)
            metrics_train.append(train_metrics)
            with torch.no_grad():
                validation_metrics = self.evaluate(valid)
                
                print(f"Train epoch #{epoch}\n"
                      f"\tloss: {train_metrics['loss']:.04f}\n"
                      f"\tacc: {train_metrics['accuracy']:.04f}\n"
                      f"\tf1: {train_metrics['f1']:.04f}\n")

                print(f"Validation epoch #{epoch}\n"
                      f"\tloss: {validation_metrics['loss']:.04f}\n"
                      f"\tacc: {validation_metrics['accuracy']:.04f}\n"
                      f"\tf1: {validation_metrics['f1']:.04f}\n")

                metrics_validation.append(validation_metrics)

        
        with torch.no_grad():
            test_metrics = self.evaluate(test)
            
            print(f"Test\n"
                      f"\tloss: {test_metrics['loss']:.04f}\n"
                      f"\tacc: {test_metrics['accuracy']:.04f}\n"
                      f"\tf1: {test_metrics['f1']:.04f}\n")

        now_str = datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
        save_file_torch(f'{SAVE_MODEL_DIR}/{TASK}', f'{SAVE_MODEL_NAME}_{now_str}.pth', self.state_dict())
        file = {'epochs': epochs, 'lr': self.lr, 'train': metrics_train, 'validation': metrics_validation, 'test': test_metrics}
        file = {**self.kwargs, **file}
        save_file_json(f'{SAVE_MODEL_DIR}/{TASK}', f'{SAVE_MODEL_NAME}_{now_str}.json', file)

    def evaluate(self, validate):    
        x, y = zip(*[(entry[0].squeeze(0), entry[1]) for entry in list(validate)])

        self.eval()

        y_pred = []
        losses = []

        for _x, _y in zip(x, y):
            y_pred.append(self.infer(_x))
            y_pred_tensor = y_pred[-1].clone().detach().float().view(_y.shape)
            losses.append(float(self.loss(_y.float(), y_pred_tensor)))

        y_pred = torch.cat(y_pred)
        y_real = torch.cat(y)

        metrics = model_metrics(y_real, y_pred)
        metrics['loss'] = float(np.mean(losses))
        return metrics