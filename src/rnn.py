import torch
import torch.nn.functional as F
import torch.nn as nn
import logging
from tokens import SOS_TOKEN

logging.basicConfig(level=logging.INFO, format="%(message)s")

log = logging.getLogger()


def as_variable(tensor, volatile=False):
    if torch.cuda.is_available():
        tensor = tensor.cuda(async=True)
    return torch.autograd.Variable(tensor, volatile=volatile)


class Seq2SeqModel(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 encoder_layers=1,
                 decoder_layers=1):
        super(Seq2SeqModel, self).__init__()

        self.hidden_size = hidden_size

        self.encoder = EncoderRNN(input_size, hidden_size, encoder_layers)
        self.decoder = DecoderRNN(hidden_size, output_size, decoder_layers)

    def forward(self, inputs, targets=None, encoder_hidden=None):
        if not encoder_hidden:
            encoder_hidden = self.init_hidden()

        encoder_output, encoder_hidden = self.encoder(inputs, encoder_hidden)

        decoder_input = targets
        decoder_hidden = encoder_hidden[-1, -1, :].repeat(
            1, targets.size(1), 1)
        log.debug("Decoder hidden [Seq2Seq]: %s", decoder_hidden.size())

        decoder_output, decoder_hidden = self.decoder(decoder_input,
                                                      decoder_hidden)
        return decoder_output, decoder_hidden

    def init_hidden(self):
        return as_variable(torch.zeros(1, 14, self.hidden_size))


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, inputs, hidden):
        log.debug("Encoder inputs: %s", inputs.size())
        log.debug("Encoder hidden: %s", hidden.size())

        embedded = self.embedding(inputs)
        log.debug("Encoder embeddings: %s", embedded.size())

        # embedded = nn.utils.rnn.pack_padded_sequence(
        #     embedded, input_lengths, batch_first=True)
        # output, hidden = self.rnn(embedded)

        output = embedded
        # for i in range(self.n_layers):
        output, hidden = self.gru(output, hidden)

        log.debug("Encoder hidden: %s", hidden.size())
        log.debug(hidden.data)
        log.debug("Encoder output: %s\n", output.size())
        # output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, inputs, hidden):
        log.debug("Decoder inputs: %s", inputs.size())
        log.debug("Decoder hidden: %s", hidden.size())
        output = self.embedding(inputs)
        log.debug("Decoder embeddings: %s", output.size())
        # for i in range(self.n_layers):

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        log.debug("Decoder RNN output: %s", output.size())

        output = self.softmax(self.out(output.squeeze(dim=1)))

        log.debug("Decoder softmax output: %s", output.size())

        output = output.squeeze(dim=1)
        return output, hidden
