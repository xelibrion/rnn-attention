#!/usr/bin/env python

import re

import pandas as pd

SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


MAX_LENGTH = 10

eng_prefixes = [
    "i am ",
    "i m ",
    "he is",
    "he s ",
    "she is",
    "she s",
    "you are",
    "you re ",
    "we are",
    "we re ",
    "they are",
    "they re ",
]


def build_filter(df):
    ft_len = (df['en_length'] < MAX_LENGTH) & (df['ru_length'] < MAX_LENGTH)

    ft_pref = df['en'].str.startswith(eng_prefixes[0])

    for prefix in eng_prefixes[1:]:
        ft_pref = ft_pref | (df['en'].str.startswith(prefix))

    return ft_len & ft_pref


def normalize_string(s):
    s = s.strip()
    # add spaces between punctunation
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Zа-яА-Я.!?]+", r" ", s)
    return s


def get_data():
    print("Reading lines...")

    df = pd.read_csv(
        '../input/rus.txt',
        sep='\t',
        header=None,
        names=['en', 'ru'], )

    for col in df.columns:
        df[col] = df[col].str.lower()

    df['en_length'] = df['en'].str.split(' ').apply(lambda x: len(x))
    df['ru_length'] = df['ru'].str.split(' ').apply(lambda x: len(x))

    df = df[build_filter(df)]

    src_lang = Lang('en')
    dst_lang = Lang('ru')

    pairs = []

    for _, en_sent, ru_sent in df[['en', 'ru']].itertuples():
        src_sent, dst_sent = normalize_string(en_sent), normalize_string(
            ru_sent)
        pairs.append((src_sent, dst_sent))
        src_lang.add_sentence(src_sent)
        dst_lang.add_sentence(dst_sent)

    return src_lang, dst_lang, pairs
