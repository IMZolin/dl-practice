import os

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizerFast

from utils import read_file, clean_text


DIRECTORY = './data/vocab.txt'
VOCAB_PATH = './data/vocab.txt'
PAD_TOKEN_IDX = 0
SEP_TOKEN_IDX = 3


def collate_batch(batch):
    src_list = []
    for src in batch:
        src_list.append(torch.tensor(src))
    return pad_sequence(src_list, padding_value=PAD_TOKEN_IDX, batch_first=True)


class DatasetMaker(torch.utils.data.Dataset):
    def __init__(self, directory, test_size, vocab_path, max_seq_len, batch_size, offset, create_test_segments=False):
        self.__paths = []
        self.__texts = []
        self.__texts_test = []
        self.load_samples(directory, test_size)

        self.__tokenizer = BertTokenizerFast(vocab_path)
        self.__max_seq_len = max_seq_len
        self.__batch_size = batch_size
        self.__offset = offset
        self.__segments = []
        self.__segments_test = []
        self.__tokenized_train = []
        self.__create_segments(create_test_segments)


    def __len__(self):
        return len(self.__segments)

    def __getitem__(self, idx):
        return self.__segments[idx]


    @property
    def tokenizer(self):
        return self.__tokenizer

    @property
    def texts(self):
        return self.__texts

    @property
    def texts_test(self):
        return self.__texts_test

    @property
    def tokenized_train(self):
        return self.__tokenized_train

    @property
    def segments_test(self):
        return self.__segments_test

    def get_texts(self):
        return self.__texts, self.__texts_test

    @staticmethod
    def __get_all_files(directory: str):
        file_paths = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.txt'):
                    file_paths.append(os.path.join(root, file))
        return file_paths

    @staticmethod
    def __load_texts(paths):
        texts = []
        for path in paths:
            text = read_file(path)
            text = clean_text(text)
            texts.append(text)
        return texts

    def load_samples(self, directory: str = "./data/books_txt", test_size = 0.03):
        self.__paths = self.__get_all_files(directory)
        if not self.__paths:
            raise ValueError(f"No text files found in the directory: {directory}")
        self.__paths.sort()
        paths_train, paths_test = train_test_split(self.__paths, test_size=test_size, random_state=12345)
        self.__texts = self.__load_texts(paths_train)
        self.__texts_test = self.__load_texts(paths_test)

    def encode(self, text):
        return self.__tokenizer.encode(text, add_special_tokens=False)

    def combine_texts(self):
        self.__tokenized_train = []
        for text in self.__texts:
            token_ids = self.encode(text)
            self.__tokenized_train.extend(token_ids)

    def __create_segments(self, create_test_segments):
        self.__segments = []
        self.__segments_test = []
        for text in self.__texts:
            tokens = self.__tokenizer.encode(text, add_special_tokens=False)
            for i in range(0, len(tokens) - self.__max_seq_len, self.__offset):
                self.__segments.append(tokens[i:i + self.__max_seq_len + 1])
                if create_test_segments:
                    self.__segments_test.append(tokens[i:i + self.__max_seq_len + 1])
