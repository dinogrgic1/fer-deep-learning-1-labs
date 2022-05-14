from dataclasses import dataclass
from email import header
from pickle import NONE
import pandas as pd
import torch

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
    texts, labels = zip(*batch) # Assuming the instance is in tuple-like form
    lengths = torch.tensor([len(text) for text in texts]) # Needed for later
    max_length = torch.max(lengths)
    texts = [torch.cat((text, torch.zeros(max_length - len(text))), 0) for text in texts]
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

    def __init__(self, file):
        input = pd.read_csv(file, names=['text', 'label'], converters={'text': lambda x: str.split(x)})
        self.instances = [Instance(text=instance.text, label=instance.label.strip()) for instance in input.itertuples()]

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        return (self.instances[index].text, self.instances[index].label)

    def from_file(file):
        dataset =  NLPDataset(file)
        text_freq, label_freq = dataset_frequencies(dataset.instances)
        dataset.text_vocab = Vocab(text_freq, max_size=-1, min_freq=0)
        dataset.label_vocab = Vocab(label_freq, max_size=-1, min_freq=0)
        dataset.instances = [Instance(text=dataset.text_vocab.encode(instance.text), label=dataset.label_vocab.encode(instance.label)) for instance in dataset.instances]
        return dataset

class Vocab:
    __SPECIAL__WORDS = ['<PAD>', '<UNK>']
    stoi = {}
    itos = []

    def __init__(self, frequencies, max_size=-1, min_freq = 0) -> None:
        words = {key:val for key, val in frequencies.items() if val >= min_freq}
        self.itos = list(words.keys())
        if max_size >= 0:
            self.itos = self.itos[:max_size]
        self.itos = self.__SPECIAL__WORDS + self.itos
        for idx, word in enumerate(self.itos):
            self.stoi[word] = idx

    def encode(self, sentence):
        if isinstance(sentence, str):
            return torch.tensor([self.stoi[sentence]])    
        return torch.tensor([self.stoi[word] for word in sentence])

    def embedding(self, data_source='file', file='dataset/sst_glove_6b_300d.txt'):
        
        if data_source == 'file':
            data = torch.empty((len(self.itos), 300))
            with open(file) as f:
                lines = f.readline().strip().split(' ')
                idx = self.encode(lines[0])
                data[idx] = torch.tensor(list(map(float, lines[1:])), dtype=torch.float32)
            return torch.nn.Embedding.from_pretrained(data, padding_idx=0)

        elif data_source == 'random':
            rand = torch.empty((len(self.itos), 300)).normal_(mean=0,std=1)
            rand[0] = torch.zeros(300)
            return torch.nn.Embedding.from_pretrained(rand, padding_idx=0)
        
        else:
            return NotImplemented
        


if __name__=='__main__':
    batch_size = 2
    shuffle = False
    train_dataset = NLPDataset.from_file('dataset/sst_train_raw.csv')
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, 
                                shuffle=shuffle, collate_fn=pad_collate_fn)
    texts, labels, lengths = next(iter(train_dataloader))
    print(f"Texts: {texts}")
    print(f"Labels: {labels}")
    print(f"Lengths: {lengths}")
