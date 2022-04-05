import torch
import json
from torch import nn
import math

import time
from pathlib import Path
import skimage as ski
import os
import matplotlib.pyplot as plt

import numpy as np
from torchvision.datasets import MNIST
from datetime import datetime

WEIGHT_DECAY = 1e-2
BATCH_SIZE = 50
EPOCHS = 8
NOW_DATE = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
DATA_DIR = Path(__file__).parent / 'datasets' / 'MNIST'
SAVE_FILTERS_DIR = Path(__file__).parent / 'out' / \
    'conv_torch' / str(WEIGHT_DECAY) / NOW_DATE / 'filters'
SAVE_MODEL_DIR = Path(__file__).parent / 'out' / \
    'conv_torch' / str(WEIGHT_DECAY) / NOW_DATE


def dense_to_one_hot(y, class_count):
    return np.eye(class_count)[y]


def draw_conv_filters(layer, file_path, file_name):
    w = layer.weight.clone().detach().numpy()
    N, C, H, W = w.shape[:4]

    layer_w_T = w.transpose(2, 3, 1, 0)
    layer_w_T -= layer_w_T.min()
    layer_w_T /= layer_w_T.max()

    num_filters = w.shape[0]
    border = 1
    cols = 8
    rows = math.ceil(num_filters / cols)
    width = cols * W + (cols-1) * border
    height = rows * H + (rows-1) * border

    for i in range(N):
        img = np.zeros([height, width, C])
        for j in range(num_filters):
            r = int(j / cols) * (W + border)
            c = int(j % cols) * (H + border)
            img[r: r + H, c: c + W, :] = layer_w_T[:, :, :, j]
    if not os.path.isdir(file_path):
        os.makedirs(file_path)
    ski.io.imsave(os.path.join(file_path, file_name), img)


class CovolutionalModel(nn.Module):
    def __init__(self, n_classes):
        super(CovolutionalModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16,
                               kernel_size=5, padding=2, dtype=torch.float, bias=True)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
                               kernel_size=5, padding=2, bias=True)

        self.maxpool1 = nn.MaxPool2d(stride=2, kernel_size=2)
        self.maxpool2 = nn.MaxPool2d(stride=2, kernel_size=2)

        self.fc1 = nn.Linear(in_features=1568, out_features=512, bias=True)
        self.fc2_logits = nn.Linear(
            in_features=512, out_features=n_classes, bias=True)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten(start_dim=1)
        self.reset_parameters()

    def get_parameters(self):
        return [self.conv1.parameters(), self.conv2.parameters(), *self.fc1.parameters()]

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        h = self.conv1(x)
        h = self.maxpool1(h)
        h = self.relu(h)

        h = self.conv2(h)
        h = self.maxpool2(h)
        h = self.relu(h)

        h = self.flatten(h)
        h = self.fc1(h)
        h = self.relu(h)
        return self.fc2_logits(h)

    def loss(self, X, Yoh_):
        return torch.mean(torch.log(torch.sum(torch.exp(X), axis=1)) - torch.sum(X * Yoh_, axis=1))

    def train(self, train_x, train_y, valid_x, valid_y, epochs, weight_decay=1e-1, batch_size=50):
        train_x_torch = torch.tensor(train_x, dtype=torch.float32)
        train_y_torch = torch.tensor(train_y, dtype=torch.float32)
        validate_x_torch = torch.tensor(valid_x, dtype=torch.float32)
        validate_y_torch = torch.tensor(valid_y, dtype=torch.float32)

        optimizer = torch.optim.SGD([{"params": [*self.conv1.parameters(), *self.conv2.parameters(), *self.fc1.parameters()], "weight_decay": weight_decay},
                                     {"params": self.fc2_logits.parameters(), "weight_decay": 0.}], lr=1e-1)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=2, gamma=0.1)

        num_examples = train_x_torch.shape[0]
        num_batches = num_examples // batch_size

        loss_train = []
        loss_validation = []

        draw_conv_filters(
            layer=self.conv1, file_path=SAVE_FILTERS_DIR, file_name=f'start_filter.png')

        for epoch in range(1, epochs + 1):
            cnt_correct = 0
            permutation_idx = np.random.permutation(num_examples)
            train_x = train_x[permutation_idx]
            train_y = train_y[permutation_idx]

            loss_avg = []
            for i in range(num_batches):
                batch_x = train_x_torch[i * batch_size:(i+1)*batch_size, :]
                batch_y = train_y_torch[i * batch_size:(i+1)*batch_size, :]
                logits = self.forward(batch_x)
                loss = self.loss(logits, batch_y)
                loss_avg.append(loss.detach().numpy())

                yp = torch.argmax(logits, 1)
                yt = torch.argmax(batch_y, 1)
                cnt_correct += (yp == yt).sum()

                loss.backward()
                optimizer.step()
                scheduler.step(epoch=epoch)

                if i % 5 == 0:
                    print("epoch %d, step %d/%d, batch loss = %.2f" %
                          (epoch, i*batch_size, num_examples, loss))
                if i > 0 and i % 50 == 0:
                    print("Train accuracy = %.2f" %
                          (cnt_correct / ((i+1)*batch_size) * 100))

                optimizer.zero_grad()

            accuracy = (cnt_correct / ((i+1)*batch_size) * 100)
            draw_conv_filters(layer=self.conv1, file_path=SAVE_FILTERS_DIR,
                              file_name=f'conv1_epoch_{epoch:02d}_acc_{accuracy:04f}.png')

            print("Train accuracy = %.2f" % (cnt_correct / num_examples * 100))
            val_loss = self.evaluate(
                validate_x_torch, validate_y_torch, batch_size)

            loss_validation.append(np.asscalar(val_loss.detach().numpy()))
            loss_train.append(np.mean(loss_avg))

        torch.save(self.state_dict(), f'{SAVE_MODEL_DIR}/model.pth')

        loss_validation = [str(x) for x in loss_validation]
        loss_train = [str(x) for x in loss_train]
        file = {"loss_validation": loss_validation, "loss_train": loss_train}
        with open(f'{SAVE_MODEL_DIR}/model.json', 'w', encoding='utf-8') as json_file:
            json.dump(file, json_file)

        loss_validation = [float(x) for x in loss_validation]
        loss_train = [float(x) for x in loss_train]
        plt.plot(loss_validation)
        plt.plot(loss_train)
        plt.show()

    def evaluate(self, x_validate, y_validate, batch_size=50):
        print("\nRunning evaluation: ")

        x_validate_torch = torch.tensor(x_validate, dtype=torch.float32)
        y_validate_torch = torch.tensor(y_validate, dtype=torch.float32)

        num_examples = x_validate.shape[0]
        assert num_examples % batch_size == 0
        num_batches = num_examples // batch_size

        cnt_correct = 0
        loss_avg = 0
        for i in range(num_batches):
            batch_x = x_validate_torch[i*batch_size:(i+1)*batch_size, :]
            batch_y = y_validate_torch[i*batch_size:(i+1)*batch_size, :]

            logits = self.forward(batch_x)
            yp = torch.argmax(logits, 1)
            yt = torch.argmax(batch_y, 1)
            cnt_correct += (yp == yt).sum()

            loss_val = self.loss(logits, batch_y)
            loss_avg += loss_val

        valid_acc = cnt_correct / num_examples * 100
        loss_avg /= num_batches
        print("accuracy = %.2f" % valid_acc)
        print("avg loss = %.2f\n" % loss_avg)

        return loss_avg


if __name__ == '__main__':
    #np.random.seed(100)
    np.random.seed(int(time.time() * 1e6) % 2**31)

    ds_train, ds_test = MNIST(DATA_DIR, train=True,
                              download=True), MNIST(DATA_DIR, train=False)
    train_x = ds_train.data.reshape(
        [-1, 1, 28, 28]).numpy().astype(np.float) / 255
    train_y = ds_train.targets.numpy()
    train_x, valid_x = train_x[:55000], train_x[55000:]
    train_y, valid_y = train_y[:55000], train_y[55000:]
    test_x = ds_test.data.reshape(
        [-1, 1, 28, 28]).numpy().astype(np.float) / 255
    test_y = ds_test.targets.numpy()
    train_mean = train_x.mean()
    train_x, valid_x, test_x = (
        x - train_mean for x in (train_x, valid_x, test_x))
    train_y, valid_y, test_y = (dense_to_one_hot(y, 10)
                                for y in (train_y, valid_y, test_y))

    conv = CovolutionalModel(10)
    conv.train(train_x, train_y, valid_x, valid_y, epochs=EPOCHS,
               batch_size=BATCH_SIZE, weight_decay=WEIGHT_DECAY)
    conv.evaluate(test_x, test_y, batch_size=BATCH_SIZE)
