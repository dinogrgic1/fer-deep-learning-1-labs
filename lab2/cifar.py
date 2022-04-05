import pickle
import torch
import json
from torch import nn
from pathlib import Path
from datetime import datetime
from sklearn.metrics import confusion_matrix

import os
import matplotlib.pyplot as plt
import numpy as np
import math

import skimage as ski
import skimage.io

DATA_DIR = 'datasets/CIFAR/'
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 50
EPOCHS = 8
NOW_DATE = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
SAVE_FILTERS_DIR = Path(__file__).parent / 'out' / 'CIFAR' / str(WEIGHT_DECAY) / NOW_DATE / 'filters'
SAVE_MODEL_DIR = Path(__file__).parent / 'out' / 'CIFAR' / str(WEIGHT_DECAY) / NOW_DATE

def draw_conv_filters(epoch, step, weights, save_dir):
    w = weights.copy()
    num_filters = w.shape[0]
    num_channels = w.shape[1]
    k = w.shape[2]
    assert w.shape[3] == w.shape[2]
    w = w.transpose(2, 3, 1, 0)
    w -= w.min()
    w /= w.max()
    border = 1
    cols = 8
    rows = math.ceil(num_filters / cols)
    width = cols * k + (cols-1) * border
    height = rows * k + (rows-1) * border
    img = np.zeros([height, width, num_channels])
    for i in range(num_filters):
        r = int(i / cols) * (k + border)
        c = int(i % cols) * (k + border)
        img[r:r+k,c:c+k,:] = w[:,:,:,i]
    filename = 'epoch_%02d_step_%06d.png' % (epoch, step)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    ski.io.imsave(os.path.join(save_dir, filename), img)

def dense_to_one_hot(y, class_count):
    return np.eye(class_count)[y]

def shuffle_data(data_x, data_y):
    indices = np.arange(data_x.shape[0])
    np.random.shuffle(indices)
    shuffled_data_x = np.ascontiguousarray(data_x[indices])
    shuffled_data_y = np.ascontiguousarray(data_y[indices])
    return shuffled_data_x, shuffled_data_y

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict

def draw_image(img, mean, std):
  img = img.transpose(1, 2, 0)
  img *= std
  img += mean
  img = img.astype(np.uint8)
  ski.io.imshow(img)
  ski.io.show()

def plot_training_progress(save_dir, data):
  fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,8))

  linewidth = 2
  legend_size = 10
  train_color = 'm'
  val_color = 'c'

  num_points = len(data['train_loss'])
  x_data = np.linspace(1, num_points, num_points)
  ax1.set_title('Cross-entropy loss')
  ax1.plot(x_data, data['train_loss'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='train')
  ax1.plot(x_data, data['valid_loss'], marker='o', color=val_color,
           linewidth=linewidth, linestyle='-', label='validation')
  ax1.legend(loc='upper right', fontsize=legend_size)
  ax2.set_title('Average class accuracy')
  ax2.plot(x_data, data['train_acc'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='train')
  ax2.plot(x_data, data['valid_acc'], marker='o', color=val_color,
           linewidth=linewidth, linestyle='-', label='validation')
  ax2.legend(loc='upper left', fontsize=legend_size)
  ax3.set_title('Learning rate')
  ax3.plot(x_data, data['lr'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='learning_rate')
  ax3.legend(loc='upper left', fontsize=legend_size)

  save_path = os.path.join(save_dir, 'training_plot.png')
  print('Plotting in: ', save_path)
  plt.savefig(save_path)



class CovolutionalModel(nn.Module):
    def __init__(self, n_classes):
        super(CovolutionalModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, padding=3, dtype=torch.float, bias=True)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=3, dtype=torch.float, bias=True)

        self.maxpool1 = nn.MaxPool2d(3, stride=2)
        self.maxpool2 = nn.MaxPool2d(3, stride=2)

        self.fc1 = nn.Linear(in_features=2048, out_features=256, bias=True)
        self.fc2 = nn.Linear(
            in_features=256, out_features=128, bias=True)
        self.fc3_logits = nn.Linear(
            in_features=128, out_features=n_classes, bias=True)

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
        h = self.relu(h)
        h = self.maxpool1(h)

        h = self.conv2(h)
        h = self.relu(h)
        h = self.maxpool2(h)

        h = self.flatten(h)
        h = self.fc1(h)
        h = self.relu(h)
        h = self.fc2(h)
        h = self.relu(h)
        return self.fc3_logits(h)

    def loss(self, X, Yoh_):
        return torch.mean(torch.log(torch.sum(torch.exp(X), axis=1)) - torch.sum(X * Yoh_, axis=1))

    def train(self, train_x, train_y, valid_x, valid_y, epochs, weight_decay=1e-1, batch_size=50):
        train_x_torch = torch.tensor(train_x, dtype=torch.float32)
        train_y_torch = torch.tensor(train_y, dtype=torch.float32)
        validate_x_torch = torch.tensor(valid_x, dtype=torch.float32)
        validate_y_torch = torch.tensor(valid_y, dtype=torch.float32)

        optimizer = torch.optim.SGD([{"params": [*self.conv1.parameters(), *self.conv2.parameters(), *self.fc1.parameters(), *self.fc2.parameters()], "weight_decay": weight_decay},
                                     {"params": self.fc3_logits.parameters(), "weight_decay": 0.}], lr=1e-1)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95, last_epoch=-1)

        num_examples = train_x_torch.shape[0]
        num_batches = num_examples // batch_size

        loss_train = []
        loss_validation = []

        acc_trains = []
        acc_validations = []

        f1_trains = []
        f1_validations = []
        lrs = []

        draw_conv_filters(0, 0, self.conv1.weight.detach().numpy(), SAVE_FILTERS_DIR)

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

                if i % 5 == 0:
                    print("epoch %d, step %d/%d, batch loss = %.2f" %
                          (epoch, i*batch_size, num_examples, loss))
                if i > 0 and i % 50 == 0:
                    print("Train accuracy = %.2f" %
                          (cnt_correct / ((i+1)*batch_size) * 100))
                optimizer.zero_grad()
            
            lrs.append(scheduler.get_last_lr()[0])
            scheduler.step()
            draw_conv_filters(epoch, 0, self.conv1.weight.detach().numpy(), SAVE_FILTERS_DIR)

            loss_avg_valid, acc_valid, precision_valid, recall_valid, f1_valid = self.evaluate(validate_x_torch, validate_y_torch, batch_size)
            loss_avg_train, acc_train, precision_train, recall_train, f1_train = self.evaluate(train_x_torch, train_y_torch, batch_size)
            
            loss_validation.append(np.asscalar(loss_avg_valid.detach().numpy()))
            loss_train.append(np.asscalar(loss_avg_train.detach().numpy()))

            acc_validations.append(np.asscalar(acc_valid.detach().numpy()))
            acc_trains.append(np.asscalar(acc_train.detach().numpy()))

            f1_validations.append(f1_valid)
            f1_trains.append(f1_train)

        torch.save(self.state_dict(), f'{SAVE_MODEL_DIR}/model.pth')
        
        loss_validation = [str(x) for x in loss_validation]
        loss_train = [str(x) for x in loss_train]
        acc_validations = [str(x) for x in acc_validations]
        acc_trains = [str(x) for x in acc_trains]
        f1_validations = [str(x) for x in f1_validations]
        f1_trains = [str(x) for x in f1_trains]
        lrs = [str(x) for x in lrs]

        file = {"valid_loss" : loss_validation, "train_loss": loss_train, "valid_acc": acc_validations, "train_acc" : acc_trains, "f1_validation": f1_validations, "f1_train": f1_trains, "lr": lrs}
        with open(f'{SAVE_MODEL_DIR}/model.json', 'w', encoding='utf-8') as json_file:
            json.dump(file, json_file)

        file["valid_loss"] = [float(x) for x in loss_validation]
        file["lr"] = [float(x) for x in lrs]
        file["train_loss"] = [float(x) for x in loss_train]
        file["valid_acc"] = [float(x) for x in acc_validations]
        file["train_acc"] = [float(x) for x in acc_trains]
        file["f1_validations"] = [float(x) for x in f1_validations]
        file["f1_trains"] = [float(x) for x in f1_trains]

        plot_training_progress(f'{SAVE_MODEL_DIR}', file)

    def evaluate(self, x_validate, y_validate, batch_size=50):
        print("\nRunning evaluation: ")

        x_validate_torch = torch.tensor(x_validate)
        y_validate_torch = torch.tensor(y_validate)
        
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
        
        cm = confusion_matrix(yt, yp)
        sums = [np.sum(cm, axis=y) for y in [None, 0, 1]]
        sums[0] = np.maximum(1, sums[0])
        for i in range(1, len(sums)):
            sums[i][sums[i] == 0] = 1

        precision, recall = [float(np.mean(np.diag(cm) / x)) for x in sums[1:]]
        f1 = (2 * precision * recall) / (precision + recall)
        print("accuracy = %.2f" % valid_acc)
        print("avg loss = %.2f\n" % loss_avg)

        return loss_avg, valid_acc, precision, recall, f1

img_height = 32
img_width = 32
num_channels = 3
num_classes = 10

train_x = np.ndarray((0, img_height * img_width * num_channels), dtype=np.float32)
train_y = []
for i in range(1, 6):
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR)
    subset = unpickle(os.path.join(DATA_DIR, 'data_batch_%d' % i))
    train_x = np.vstack((train_x, subset['data']))
    train_y += subset['labels']
train_x = train_x.reshape(
(-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1)
train_y = np.array(train_y, dtype=np.int32)

subset = unpickle(os.path.join(DATA_DIR, 'test_batch'))
test_x = subset['data'].reshape(
(-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1).astype(np.float32)
test_y = np.array(subset['labels'], dtype=np.int32)

valid_size = 5000
train_x, train_y = shuffle_data(train_x, train_y)
valid_x = train_x[:valid_size, ...]
valid_y = train_y[:valid_size, ...]
train_x = train_x[valid_size:, ...]
train_y = train_y[valid_size:, ...]
data_mean = train_x.mean((0, 1, 2))
data_std = train_x.std((0, 1, 2))

train_x = (train_x - data_mean) / data_std
valid_x = (valid_x - data_mean) / data_std
test_x = (test_x - data_mean) / data_std

train_x = train_x.transpose(0, 3, 1, 2)
valid_x = valid_x.transpose(0, 3, 1, 2)
test_x = test_x.transpose(0, 3, 1, 2)

train_y = dense_to_one_hot(train_y, 10)
valid_y = dense_to_one_hot(valid_y, 10)
test_y = dense_to_one_hot(test_y, 10)

conv = CovolutionalModel(num_classes)
conv.train(train_x, train_y, valid_x, valid_y, epochs=EPOCHS, batch_size=BATCH_SIZE, weight_decay=WEIGHT_DECAY)
conv.evaluate(test_x, test_y, batch_size=BATCH_SIZE)
