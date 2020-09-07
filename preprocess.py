import pandas as pd
import unicodedata
import re 
import numpy as np 

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = w.rstrip().strip()
    return w

def hindi_preprocess_sentence(w):
    w = w.rstrip().strip()
    return w

def split_df(df_path, save_path, train_percent=.8, validate_percent=.1, seed=None):
    df = pd.read_csv(df_path)
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    
    test = df.iloc[perm[validate_end:]]
    
    train.to_csv(save_path + 'train_df.csv',index = False)
    test.to_csv(save_path + 'test_df.csv',index = False)
    validate.to_csv(save_path + 'validate_df.csv',index = False)
    return True

def preprocess(df_path, save_path):
    df = pd.read_csv(df_path)

    df=df.dropna()
    df = df[df['source']=='ted']

    df.english_sentence = df.english_sentence.apply(lambda x: ' '.join([preprocess_sentence(w) for w in  x.split(' ')]))

    df.hindi_sentence = df.hindi_sentence.apply(lambda x: ' '.join([hindi_preprocess_sentence(w) for w in  x.split(' ')]))
    df.iloc[:,1:].to_csv(save_path + 'preprcessed30K.csv',index = False)
    #df = df.iloc[:,1:]

    splitted = split_df(save_path + 'preprcessed30K.csv', save_path)

    if splitted:
        return "Splitting Done! Saved"
    else:
        return "Error while Splitting"

preprocess('Language-Translation-Using-PyTorch/input/enghin.csv', 'Language-Translation-Using-PyTorch/input/')


