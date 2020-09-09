from dataset import TranslateDataset
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

class MyCollate:
    def __init__(self,padIdx):
        self.padIdx = padIdx

    def __call__(self,batch):
        sources = [item[0] for item in batch]       
        sources = pad_sequence(sources, batch_first=False, padding_value= self.padIdx)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value= self.padIdx)

        return sources,targets 


def get_loader(root_path, batch_size = 32,shuffle = True, num_workers = 8,pin_memory = True):
    trainDataset = TranslateDataset(root_path + 'train_df.csv')
    valDataset = TranslateDataset(root_path + 'validate_df.csv')
    testDataset = TranslateDataset(root_path + 'test_df.csv')
    
    trainDataset.eng_vocab.storeVocab('eng')
    trainDataset.hin_vocab.storeVocab('hin')
    padIdx = trainDataset.eng_vocab.stoi['<PAD>']
    
    trainLoader = DataLoader(
        dataset =trainDataset,
        batch_size= batch_size,
        shuffle = shuffle, 
        num_workers = num_workers,
        pin_memory = pin_memory,
        collate_fn = MyCollate(padIdx = padIdx )
    )

    valLoader = DataLoader(
        dataset = valDataset,
        batch_size= batch_size,
        shuffle = shuffle, 
        num_workers = num_workers,
        pin_memory = pin_memory,
        collate_fn = MyCollate(padIdx = padIdx )
    )

    testLoader = DataLoader(
        dataset = testDataset,
        batch_size= batch_size,
        shuffle = shuffle, 
        num_workers = num_workers,
        pin_memory = pin_memory,
        collate_fn = MyCollate(padIdx = padIdx )
    )

    return trainLoader, valLoader, testLoader, trainDataset

# trainLoader, valLoader, testLoader, trainDataset = get_loader(root_path = 'Language-Translation-Using-PyTorch/input/')

# for idx, (src, trg) in enumerate(trainLoader):
#     print(src.shape)
#     print(trg.shape)
#     print(src.permute(1,0).shape)
#     print(trg.permute(1,0).shape)
#     for src_text, trg_text in zip(src.permute(1,0),trg.permute(1,0)):
#         print(src_text.shape)
#         print(trg_text.shape)
#         break
#     break


