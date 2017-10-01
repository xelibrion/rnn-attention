#!/usr/bin/env python

import argparse
import os
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

from model_tuner import Tuner
from preprocess import get_pairs, to_numpy_tensor_pair
from rnn import Seq2SeqModel
import logging


def define_args():

    parser = argparse.ArgumentParser(description='Time series via neural nets')

    parser.add_argument(
        '-j',
        '--workers',
        default=4,
        type=int,
        metavar='N',
        help='number of data loading workers (default: 4)')

    parser.add_argument(
        '-d',
        '--debug',
        action='store_true',
        help='enables debug mode', )

    parser.add_argument(
        '--epochs',
        default=90,
        type=int,
        metavar='N',
        help='number of total epochs to run')

    parser.add_argument(
        '--start-epoch',
        default=0,
        type=int,
        metavar='N',
        help='manual epoch number (useful on restarts)')

    parser.add_argument(
        '--epoch-size',
        default=None,
        type=int,
        metavar='N',
        help='manual epoch size (useful for debugging)')

    parser.add_argument(
        '-b',
        '--batch-size',
        default=72,
        type=int,
        metavar='N',
        help='mini-batch size (default: 72)')

    parser.add_argument(
        '--lr',
        '--learning-rate',
        default=1e-3,
        type=float,
        metavar='LR',
        help='initial learning rate')

    parser.add_argument(
        '--resume',
        default='',
        type=str,
        metavar='PATH',
        help='path to latest checkpoint (default: none)')

    return parser


HIDDEN_SIZE = 256


def create_model(in_vocabulary_size, out_vocabulary_size):
    model = Seq2SeqModel(
        in_vocabulary_size,
        HIDDEN_SIZE,
        out_vocabulary_size, )
    return model, model.encoder.parameters(), model.decoder.parameters()


def create_data_pipeline(args):
    print("Loading data")

    if Path('./data.pkl').exists():
        in_lang, out_lang, pairs = joblib.load('./data.pkl')
    else:
        in_lang, out_lang, pairs = get_pairs()
        joblib.dump((in_lang, out_lang, pairs), './data.pkl')

    X, y = to_numpy_tensor_pair(in_lang, out_lang, pairs)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=.2,
        random_state=42, )

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    train_set = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train), )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False, )
    val_set = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test), )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False, )

    return train_loader, val_loader, in_lang, out_lang


def main():
    parser = define_args()
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format="%(message)s")

    cudnn.benchmark = True

    train_loader, val_loader, in_lang, out_lang = create_data_pipeline(args)

    model, encoder_params, decoder_params = create_model(
        in_lang.n_words, out_lang.n_words)
    criterion = torch.nn.NLLLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        # decoder = decoder.cuda()
        criterion = criterion.cuda()

    encoder_optimizer = torch.optim.Adam(encoder_params, args.lr)
    decoder_optimizer = torch.optim.Adam(decoder_params, args.lr)

    tuner = Tuner(
        model,
        encoder_optimizer,
        decoder_optimizer,
        criterion,
        max_length=14, )

    if args.resume:
        if os.path.isfile(args.resume):
            tuner.restore_checkpoint(args.resume)

    tuner.run(train_loader, val_loader)


if __name__ == '__main__':
    main()
