#!/usr/bin/env python

import argparse
import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data

from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from pathlib import Path
import quandl

from model_tuner import Tuner
from time_series import RNN


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


def create_model(initial_lr):
    model = RNN()
    return model, model.parameters()


def create_data_pipeline(args):
    print("Loading data")

    if Path('./AMD.pkl').exists():
        data_df = joblib.load('./AMD.pkl')
    else:
        data_df = quandl.get("WIKI/AMD")
        joblib.dump(data_df, './AMD.pkl')

    data_df = data_df[[x for x in data_df.columns if 'adj' in x.lower()]]
    data_df.columns = [
        x.lower().replace('.', '').replace(' ', '_') for x in data_df.columns
    ]

    scaler = MinMaxScaler()

    close = data_df.loc['2014-01-01':, 'adj_close']

    df = data_df.loc['2014-01-01':, :].copy()
    df['y'] = close.diff(-5).dropna()
    df['y'] = (df['y'] > 0.0).astype(int)
    df.dropna(inplace=True)

    X = scaler.fit_transform(df.iloc[:, :-1])

    X_train = X[:-50, :]
    X_test = X[-50:, :]
    y_train = df['y'].values[:-50]
    y_test = df['y'].values[-50:]

    train_set = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train).float().view(1, len(X_train), -1),
        torch.from_numpy(y_train).float().view(1, len(y_train), 1))
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=5,
        shuffle=False, )
    val_set = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test).float().view(1, len(X_test), -1),
        torch.from_numpy(y_test).float().view(1, len(y_test), 1))
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False, )

    return train_loader, val_loader


def main():
    parser = define_args()
    args = parser.parse_args()

    cudnn.benchmark = True

    model, full_params = create_model(args.lr)
    criterion = torch.nn.BCELoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    optimizer = torch.optim.Adam(full_params, args.lr)

    train_loader, val_loader = create_data_pipeline(args)

    tuner = Tuner(
        model,
        criterion,
        optimizer, )
    if args.resume:
        if os.path.isfile(args.resume):
            tuner.restore_checkpoint(args.resume)

    tuner.run(train_loader, val_loader)


if __name__ == '__main__':
    main()
