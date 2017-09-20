import collections
import json
import os
import shutil
import time
from datetime import datetime

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


def as_variable(tensor, volatile=False):
    if torch.cuda.is_available():
        tensor = tensor.cuda(async=True)
    return torch.autograd.Variable(tensor, volatile=volatile)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, window_size=20):
        self.reset(window_size)

    def reset(self, window_size):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.window = collections.deque([], window_size)

    @property
    def mavg(self):
        return np.mean(self.window)

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.window.append(self.val)


class Emitter:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def __call__(self, event):
        with open(self.path, 'a') as out_file:
            event.update({'timestamp': datetime.utcnow().isoformat()})
            out_file.write(json.dumps(event))
            out_file.write('\n')


class Tuner:
    def __init__(self,
                 encoder,
                 decoder,
                 encoder_optimizer,
                 decoder_optimizer,
                 criterion,
                 max_length,
                 epochs=200,
                 early_stopping=None,
                 tag=None):
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
        self.criterion = criterion
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.start_epoch = 0
        self.max_length = max_length
        self.best_score = -float('Inf')
        self.tag = tag
        self.h_state = None
        self.emit = Emitter('./logs/events.json' if not tag else
                            './logs/events_{}.json'.format(tag))

    def restore_checkpoint(self, checkpoint_file):
        print("=> loading checkpoint '{}'".format(checkpoint_file))

        checkpoint = torch.load(checkpoint_file)
        self.start_epoch = checkpoint['epoch']
        self.best_score = checkpoint['best_score']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("=> loaded checkpoint '{}' (epoch {})"
              .format(checkpoint_file, checkpoint['epoch']))

    def save_checkpoint(self, validation_score, epoch):
        checkpoint_filename = ('checkpoint.pth.tar' if not self.tag else
                               'checkpoint_{}.pth.tar'.format(self.tag))
        best_model_filename = ('model_best.pth.tar' if not self.tag else
                               'model_best_{}.pth.tar'.format(self.tag))

        is_best = validation_score > self.best_score
        self.best_score = max(validation_score, self.best_score)

        state = {
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'best_score': self.best_score,
            'optimizer': self.optimizer.state_dict(),
        }

        torch.save(state, checkpoint_filename)
        if is_best:
            shutil.copyfile(checkpoint_filename, best_model_filename)

    def run(self, train_loader, val_loader):
        self.train_nnet(train_loader, val_loader)

    def train_nnet(self, train_loader, val_loader):

        for epoch in range(self.start_epoch, self.epochs):
            self.train_epoch(
                train_loader,
                epoch,
                'training',
                'Epoch #{epoch}', )

            # val_score = self.validate(
            #     val_loader,
            #     epoch,
            #     'validation',
            #     'Validating #{epoch}', )

            # scheduler.step(val_score)

            # if self.early_stopping:
            #     if self.early_stopping.should_trigger(
            #             epoch,
            #             val_score, ):
            #         break

            # self.save_checkpoint(val_score, epoch)

    def train_epoch(self, train_loader, epoch, stage, format_str):
        batch_time = AverageMeter()
        losses = AverageMeter()

        self.encoder.train()
        self.decoder.train()

        tq = tqdm(total=len(train_loader))
        description = format_str.format(**locals())
        tq.set_description('{:16}'.format(description))

        batch_idx = -1
        end = time.time()

        for i, (inputs, target) in enumerate(train_loader):
            batch_idx += 1

            loss = 0

            input_length = inputs.size(1)
            target_length = target.size(1)

            input_var = as_variable(inputs)
            target_var = as_variable(target)

            encoder_hidden = as_variable(self.encoder.init_hidden())
            encoder_outputs = as_variable(
                torch.zeros(self.max_length, self.encoder.hidden_size))

            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(
                    input_var[0, ei], encoder_hidden)
                encoder_outputs[ei] = encoder_output[0][0]

            SOS_TOKEN = 0
            EOS_TOKEN = 1

            decoder_input = as_variable(torch.LongTensor([[SOS_TOKEN]]))
            decoder_hidden = encoder_hidden

            for di in range(target_length):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input,
                    decoder_hidden, )
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]

                decoder_input = as_variable(torch.LongTensor([[ni]]))

                loss += self.criterion(decoder_output, target_var[0, di])
                if ni == EOS_TOKEN:
                    break

            batch_size = inputs.size(0)
            losses.update(loss.data[0], batch_size)

            self.decoder_optimizer.zero_grad()
            self.encoder_optimizer.zero_grad()

            loss.backward()

            self.decoder_optimizer.step()
            self.encoder_optimizer.step()

            batch_time.update(time.time() - end)

            tq.set_postfix(
                batch_time='{:.3f}'.format(batch_time.mavg),
                loss='{:.3f}'.format(losses.mavg), )
            tq.update()

            self.emit({
                'stage': stage,
                'epoch': epoch,
                'batch': batch_idx,
                'loss': losses.val
            })

            end = time.time()

        tq.close()

    def validate(self, val_loader, epoch, stage, format_str):
        batch_time = AverageMeter()
        losses = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        tq = tqdm(total=len(val_loader))
        description = format_str.format(**locals())
        tq.set_description('{:16}'.format(description))

        batch_idx = -1
        end = time.time()
        for i, (inputs, target) in enumerate(val_loader):
            batch_idx += 1

            input_var = as_variable(inputs, volatile=True)
            target_var = as_variable(target, volatile=True)

            output, h_state = self.model(input_var, self.h_state)
            loss = self.criterion(output, target_var)

            batch_size = inputs.size(0)
            losses.update(loss.data[0], batch_size)

            batch_time.update(time.time() - end)

            tq.set_postfix(
                batch_time='{:.3f}'.format(batch_time.mavg),
                loss='{:.3f}'.format(losses.mavg), )
            tq.update()

            self.emit({
                'stage': stage,
                'epoch': epoch,
                'batch': batch_idx,
                'loss': losses.val
            })
            end = time.time()

        tq.close()

        print('Validation results (avg): loss = {:.3f}\n'.format(losses.avg))
        return losses.avg
