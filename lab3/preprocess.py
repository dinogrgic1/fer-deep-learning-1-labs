from dataclasses import dataclass
import pandas as pd
import torch
import numpy as np
import random

from rnn import RNNModel

def key_add_one(dict, key):
    if key not in dict:
        dict[key] = 0
    dict[key] += 1

def dataset_frequencies(instances):
    freq_vocab = {}
    freq_label = {}

    for instance in instances:
        [key_add_one(freq_vocab, word) for word in instance.text]
        key_add_one(freq_label, instance.label)

    freq_vocab = dict(sorted(freq_vocab.items(), key=lambda item: item[1], reverse=True))
    freq_label = dict(sorted(freq_label.items(), key=lambda item: item[1], reverse=True))
    return freq_vocab, freq_label

def pad_collate_fn(batch, pad_index=0):
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(text) for text in texts])
    max_length = torch.max(lengths)

    texts = torch.tensor(np.array([np.concatenate((text, np.zeros(max_length - len(text), dtype=np.int32)), pad_index) for text in texts]))
    labels = torch.tensor([label.numpy()[0] for label in labels], dtype=torch.float32)
    return texts, labels, lengths

@dataclass
class Instance:
    text = None
    label = None

    def __init__(self, text, label):
        self.text = text
        self.label = label

    def __iter__(self):
        return iter((self.text, self.label))


class NLPDataset(torch.utils.data.Dataset):
    instances = []
    text_vocab = None
    label_vocab = None

    def __init__(self, file, text_vocab=None, label_vocab=None):
        input = pd.read_csv(file, names=['text', 'label'], converters={'text': lambda x: str.split(x)})
        self.instances = [Instance(text=instance.text, label=instance.label.strip()) for instance in input.itertuples()]
        self.text_vocab = text_vocab
        self.label_vocab = label_vocab

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        return (self.instances[index].text, self.instances[index].label)

    def from_file(file, max_size=-1, min_freq=0, text_vocab=None, label_vocab=None):
        dataset =  NLPDataset(file, text_vocab, label_vocab)
        text_freq, label_freq = dataset_frequencies(dataset.instances)
        if text_vocab != None:
            dataset.text_vocab = text_vocab
        else:
            dataset.text_vocab = Vocab(text_freq, max_size, min_freq)

        if label_vocab != None:
            dataset.label_vocab = label_vocab
        else:
            dataset.label_vocab = Vocab(label_freq, max_size, min_freq, label=True)

        dataset.instances = [Instance(text=dataset.text_vocab.encode(instance.text), label=dataset.label_vocab.encode(instance.label)) for instance in dataset.instances]
        return dataset

class Vocab:
    __SPECIAL__WORDS = ['<PAD>', '<UNK>']
    stoi = {}
    itos = []

    def __init__(self, frequencies, max_size=-1, min_freq=0, label=False) -> None:
        words = {key:val for key, val in frequencies.items() if val >= min_freq}
        self.itos = list(words.keys())
        
        if max_size >= 0:
            self.itos = self.itos[:max_size]
        
        if label == False:
            self.itos = self.__SPECIAL__WORDS + self.itos
        
        for idx, word in enumerate(self.itos):
            self.stoi[word] = idx

    def encode(self, sentence):
        if isinstance(sentence, str):
            if sentence not in self.stoi:
                return torch.tensor([self.stoi['<UNK>']], dtype=torch.long)    
            else:
                return torch.tensor([self.stoi[sentence]], dtype=torch.long)   

        arr = []
        for word in sentence:
            if word in self.stoi:
                arr.append(self.stoi[word])
            else:
                arr.append(self.stoi['<UNK>'])
        return torch.tensor(arr, dtype=torch.long)

    def embedding(self, data_source='file', file='dataset/sst_glove_6b_300d.txt'):
        rand = torch.empty((len(self.itos), 300)).normal_(mean=0, std=1)
        rand[0] = torch.zeros(300)

        if data_source == 'file':
            data = torch.empty((len(self.itos), 300))
            
            with open(file) as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip().split(' ')
                    idx = self.encode(line[0])
                    data[idx] = torch.tensor(list(map(float, line[1:])), dtype=torch.float32)
            return torch.nn.Embedding.from_pretrained(data, padding_idx=0, freeze=True)
        
        else:
            return torch.nn.Embedding.from_pretrained(data, padding_idx=0, freeze=False)



if __name__=='__main__':
    

    train_dataset = NLPDataset.from_file('dataset/sst_train_raw.csv', min_freq=1, max_size=-1)
    embedding_matrix_train = train_dataset.text_vocab.embedding()
    test_dataset = NLPDataset.from_file('dataset/sst_test_raw.csv', min_freq=1, max_size=-1, text_vocab=train_dataset.text_vocab, label_vocab=train_dataset.label_vocab)
    valid_dataset = NLPDataset.from_file('dataset/sst_valid_raw.csv', min_freq=1, max_size=-1, text_vocab=train_dataset.text_vocab, label_vocab=train_dataset.label_vocab)
    
    train = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True, batch_size=10, collate_fn=pad_collate_fn)
    test = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=True, batch_size=32, collate_fn=pad_collate_fn)
    valid = torch.utils.data.DataLoader(dataset=valid_dataset, shuffle=True, batch_size=32, collate_fn=pad_collate_fn)

    # for i in range(0, 5):
    #     seed = random.getrandbits(32)
    #     torch.manual_seed(seed)
    #     np.random.seed(seed)
    #     model = BaselineModel(embedding_matrix_train, torch.optim.Adam, lr=1e-3, seed=seed, test_number=i)
    #     model.fit(train, valid, test, 5)

    # for i in range(0, 5):
    #     seed = random.getrandbits(32)
    #     torch.manual_seed(seed)
    #     np.random.seed(seed)
    #     model = LSTMModel(embedding_matrix_train, torch.optim.Adam, lr=1e-4, seed=seed, test_number=i)
    #     model.fit(train, valid, test, 5)

    combination_1 = (random.randrange(100, 601), random.randrange(2, 10), random.uniform(0.1, 0.9))
    combination_2 = (random.randrange(100, 601), random.randrange(2, 10), random.uniform(0.1, 0.9))
    combination_3 = (random.randrange(100, 601), random.randrange(2, 10), random.uniform(0.1, 0.9))
    
    for cell in [torch.nn.RNN, torch.nn.GRU, torch.nn.LSTM]:
        for hidden_size, num_layers, dropout in (combination_1, combination_2, combination_3):
            
            seed = random.getrandbits(32)
            torch.manual_seed(seed)
            np.random.seed(seed)

            name = f'{str(cell)}_{hidden_size}_{num_layers}_{dropout}'
            print(f'MODEL: {name}\n')
            model = RNNModel(embedding_matrix_train, torch.optim.Adam, rnn=cell, lr=1e-4, seed=seed, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, name=name)
            model.fit(train, valid, test, 5)