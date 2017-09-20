#!/usr/bin/env python
import re
import numpy as np
import pandas as pd

SOS_TOKEN = 0
EOS_TOKEN = 1


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

    def __getitem__(self, key):
        return self.word2index[key]


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
    s = re.sub(r"[^a-zа-я.!?]+", r" ", s)
    return s


def get_pairs():
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

    in_lang = Lang('en')
    out_lang = Lang('ru')

    pairs = []

    for _, en_sent, ru_sent in df[['en', 'ru']].itertuples():
        in_sent, out_sent = normalize_string(en_sent), normalize_string(
            ru_sent)
        pairs.append((in_sent, out_sent))
        in_lang.add_sentence(in_sent)
        out_lang.add_sentence(out_sent)

    return in_lang, out_lang, pairs


def sentence_to_indices(sentence, lang):
    return [lang[word] for word in sentence.split(' ')] + [EOS_TOKEN]


def pad_sequence(sequence):
    # sequence.sort(key=lambda x: len(x), reverse=True)
    lengths = [len(x) for x in sequence]
    max_length = max(lengths)

    for seq in sequence:
        pad_length = max_length - len(seq)
        if pad_length:
            seq[:] = seq + [0] * pad_length
    return sequence, lengths


def to_numpy_tensor_pair(in_lang, out_lang, pairs):
    in_seq = [sentence_to_indices(in_text, in_lang) for in_text, _ in pairs]
    in_padded_seq, in_lengths = pad_sequence(in_seq)

    out_seq = [
        sentence_to_indices(out_text, out_lang) for _, out_text in pairs
    ]
    out_padded_seq, out_lengths = pad_sequence(out_seq)

    return np.array(in_padded_seq), np.array(out_padded_seq)
