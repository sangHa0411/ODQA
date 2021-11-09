import os
import pandas as pd

def load_vocab(dir_path, file_name) :
    assert file_name.endswith('.txt')
    vocab_map = {}
    idx = 0
    file_path = os.path.join(dir_path, file_name)

    f = open(file_path, 'r')
    while True:
        line = f.readline()
        if not line: 
            break
        vocab = line[:-1]
        vocab_map[idx] = vocab
        idx += 1

    f.close()
    return vocab_map

def add_unused(vocab_map, unused_start, vocab_size, unk_chacters) :
    assert unk_chacters.endswith('.csv')
    unk_ch_df = pd.read_csv(unk_chacters)
    
    unused_size = vocab_size - unused_start 
    for i in range(unused_size) :
        unused_idx = unused_start + i
        data = unk_ch_df.iloc[i]
        unk_ch = data['Character']
        vocab_map[unused_idx] = unk_ch

def write_vocab_txt(vocab_map, file_path) :
    assert file_path.endswith('.txt')
    data_size = len(vocab_map)
    vocab_list = list(vocab_map.values())

    f = open(file_path, 'w')
    for i in range(data_size):
        f.write(vocab_list[i]+'\n')
    f.close()