import os
import json
import pandas as pd
from transformers import PreTrainedTokenizerFast

class TokenizerOptimization :
    def __init__(self, dir_path, unk_vocab_path) :
        self.txt_path = os.path.join(dir_path, 'vocab.txt')
        self.json_path = os.path.join(dir_path, 'tokenizer.json')
        self.unk_vocab_path = unk_vocab_path

    def load_vocab(self) :
        vocab_map = {}
        idx = 0
        f = open(self.txt_path, 'r')
        while True:
            line = f.readline()
            if not line: 
                break
            vocab = line[:-1]
            vocab_map[idx] = vocab
            idx += 1
        f.close()
        return vocab_map

    def write_vocab(self, vocab_map) :
        data_size = len(vocab_map)
        vocab_list = list(vocab_map.values())
        f = open(self.txt_path, 'w')
        for i in range(data_size):
            f.write(vocab_list[i]+'\n')
        f.close()

    def add_unused(self, vocab_map, tokenizer) :
        assert self.unk_vocab_path.endswith('.csv')
        vocab_size = len(tokenizer)
        unused_start = tokenizer.convert_tokens_to_ids('[unused0]')

        unk_ch_df = pd.read_csv(self.unk_vocab_path)
        unused_size = vocab_size - unused_start 
        for i in range(unused_size) :
            unused_idx = unused_start + i
            data = unk_ch_df.iloc[i]
            unk_ch = data['Token']
            vocab_map[unused_idx] = unk_ch

    def load_tokenizer_json(self) :
        with open(self.json_path, "r") as json_data:
            tokenizer_data = json.load(json_data)
        return tokenizer_data

    def write_tokenizer_json(self, tokenizer_data, vocab_data) :
        inverse_vocab_data = {vocab_data[key] : key for key in vocab_data.keys()}
        tokenizer_data['model']['vocab'] = inverse_vocab_data
        with open(self.json_path, 'w') as json_file:
            json.dump(tokenizer_data, json_file)

    def optimize(self, tokenizer) :
        assert isinstance(tokenizer, PreTrainedTokenizerFast)
        vocab_map = self.load_vocab()
        self.add_unused(vocab_map, tokenizer)
        self.write_vocab(vocab_map)

        tokenizer_data = self.load_tokenizer_json()
        self.write_tokenizer_json(tokenizer_data, vocab_map)

