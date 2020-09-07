import torch 
from torch.utils.data import DataLoader,Dataset
import pandas as pd 
import spacy 
from torch.nn.utils.rnn import pad_sequence
import os 
import pickle 
import unicodedata

class Vocabulary:
    def __init__(self, freqThresold):
        self.itos = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.stoi = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.freqThresold = freqThresold

    def __len__(self):
        return len(self.stoi)

    @staticmethod
    def tokenizer(text):
        return [tok.lower() for tok in text.split(" ")]

    def build_vocabulary(self,sentenceList):
        freq = {}

        idx = 4

        for sent in sentenceList:
            for word in self.tokenizer(sent):
                #print(word)
                if word not in freq:
                    freq[word] = 1
                else:
                    freq[word] += 1

                if freq[word] == self.freqThresold:
                    self.itos[idx] = word
                    self.stoi[word] = idx 
                    idx += 1

    def encode(self,text):
        tokenizedText = self.tokenizer(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi['<UNK>'] for token in tokenizedText
        ]

    def storeVocab(self,name):
        print("Saving Vocab Dict...")
        with open('Language-Translation-Using-PyTorch/output/' + name+ '_itos.pkl', 'wb') as f:
            pickle.dump(self.itos, f, pickle.HIGHEST_PROTOCOL)

        with open('Language-Translation-Using-PyTorch/output/' + name + '_stoi.pkl', 'wb') as f:
            pickle.dump(self.stoi, f, pickle.HIGHEST_PROTOCOL)

class TranslateDataset(Dataset):
    def __init__(self,text_file, freqThresold = 2):
        self.df = pd.read_csv(text_file)

        self.english = self.df['english_sentence']
        self.hindi = self.df['hindi_sentence']

        self.eng_vocab = Vocabulary(freqThresold)
        self.hin_vocab = Vocabulary(freqThresold)
        self.eng_vocab.build_vocabulary(self.english.tolist())
        self.hin_vocab.build_vocabulary(self.hindi.tolist())


    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        english = self.english[index]
        hindi = self.hindi[index]

        encoded_english = [self.eng_vocab.stoi["<SOS>"]] 
        encoded_english += self.eng_vocab.encode(english)
        encoded_english.append(self.eng_vocab.stoi["<EOS>"])

        encoded_hindi = [self.hin_vocab.stoi["<SOS>"]] 
        encoded_hindi += self.hin_vocab.encode(hindi)
        encoded_hindi.append(self.hin_vocab.stoi["<EOS>"])

        return torch.tensor(encoded_english), torch.tensor(encoded_hindi)

"""
class MyCollate:
    def __init__(self,padIdx):
        self.padIdx = padIdx

    def __call__(self,batch):
        imgs = [item[0].unsqueeze(0) for item in batch]       
        imgs = torch.cat(imgs,dim = 0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value= self.padIdx)

        return imgs,targets 

"""


# trainDataset = TranslateDataset('Language-Translation-Using-PyTorch/input/train_df.csv')

# print(trainDataset.eng_vocab.stoi.keys())


