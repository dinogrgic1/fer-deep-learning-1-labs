import torch
from torch import nn
import numpy as np
from datetime import datetime

from utils import save_file_json, save_file_torch, model_metrics

SAVE_MODEL_DIR = 'saved'
TASK = 'task_4'

class RNNModel(nn.Module):
    def __init__(self, embedding_matrix, optimizer=torch.optim.Adam, lr=1e-4, hidden_size=150, rnn=torch.nn.LSTM, num_layers=2, dropout=0., gradient_clipping=0.2, bidirectional=False, name='model', *args, **kwargs):
        super().__init__()

        self.name = name
        
        self.rnn1 = rnn(300, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=False)
        self.rnn2 = rnn(hidden_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=False)
        
        input_fc = hidden_size
        if bidirectional:
            input_fc *= 2

        self.fc1 = torch.nn.Linear(input_fc, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, 1)
        self.loss =  torch.nn.BCEWithLogitsLoss()
        
        self.kwargs = kwargs
        self.embedding_matrix = embedding_matrix
        self.gradient_clipping = gradient_clipping
        self.lr = lr
        self.optimizer = optimizer(self.get_parameters(), lr=self.lr)

        self.reset_parameters()

    def get_parameters(self):
        return [*self.fc1.parameters(), *self.fc2.parameters(), *self.rnn2.parameters(), *self.rnn1.parameters(), *self.embedding_matrix.parameters()]

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.normal_(m.bias, 0, 1e-6 / 3)



    def forward(self, x):
        y = self.embedding_matrix(x)
        y = torch.transpose(y, 0, 1)

        y, h = self.rnn1(y, None)
        y, h = self.rnn2(y, h)

        y = y[-1]

        y = self.fc1(y)
        y = torch.relu(y)

        return self.fc2(y)

    def infer(self, x) -> int:
        with torch.no_grad():
            y = self.forward(x)
            y = torch.sigmoid(y)
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

                if self.gradient_clipping is not None:
                    torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.gradient_clipping)
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
        save_file_torch(f'{SAVE_MODEL_DIR}/{TASK}', f'{self.name}_{now_str}.pth', self.state_dict())
        file = {'epochs': epochs, 'lr': self.lr, 'train': metrics_train, 'validation': metrics_validation, 'test': test_metrics}
        file = {**self.kwargs, **file}
        save_file_json(f'{SAVE_MODEL_DIR}/{TASK}', f'{self.name}_{now_str}.json', file)

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