import torch
import torch.nn.functional as F
import torch.nn as nn
import logging
from tokens import SOS_TOKEN, PAD_TOKEN

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
            encoder_hidden = self.init_hidden(inputs.size(0))

        encoder_output, encoder_hidden = self.encoder(inputs, encoder_hidden)

        decoder_input = targets
        decoder_hidden = encoder_hidden[:, -1:, :]
        log.debug("Decoder hidden [Seq2Seq]: %s", decoder_hidden.size())

        decoder_output, decoder_hidden = self.decoder(decoder_hidden,
                                                      inputs.size(0),
                                                      targets.size(1))
        return decoder_output, decoder_hidden

    def init_hidden(self, batch_size):
        return as_variable(torch.zeros(1, batch_size, self.hidden_size))


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

        lengths = inputs.data.ne(PAD_TOKEN).long().sum(1).squeeze()
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths = list(lengths[idx_sort])
        idx_sort = as_variable(idx_sort)
        idx_unsort = as_variable(idx_unsort)

        embedded = self.embedding(inputs)
        log.debug("Encoder embeddings: %s", embedded.size())

        # Sort x
        inputs_sorted = embedded.index_select(0, idx_sort)
        # (B,L,D) -> (L,B,D)
        inputs_sorted = inputs_sorted.transpose(0, 1)

        # Pack it up
        packed = nn.utils.rnn.pack_padded_sequence(inputs_sorted, lengths)
        packed_out, packed_hidden = self.gru(packed, hidden)

        output, _ = nn.utils.rnn.pad_packed_sequence(packed_out)

        hidden = packed_hidden
        log.debug("Encoder hidden: %s", packed_hidden.size())
        log.debug(hidden.data)
        log.debug("Encoder output: %s\n", output.size())

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

    def _forward_step(self, inputs, hidden):

        log.debug("Decoder _forward_step inputs: %s", inputs.size())
        log.debug("Decoder _forward_step hidden: %s", hidden.size())

        output = self.embedding(inputs)
        log.debug("Decoder _forward_step embeddings: %s", output.size())

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        log.debug("Decoder _forward_step RNN output: %s", output.size())

        output = self.softmax(self.out(output.squeeze(dim=1)))

        log.debug("Decoder _forward_step softmax output: %s", output.size())

        output = output.squeeze(dim=1)
        log.debug("Decoder _forward_step output: %s", output.size())

        return output, hidden

    def forward(self, hidden, batch_size, target_length):

        inputs = torch.LongTensor([SOS_TOKEN]).repeat(batch_size, 1)
        inputs = as_variable(inputs)

        outputs = []

        for step in range(target_length):
            output, hidden = self._forward_step(inputs, hidden)

            outputs.append(output.unsqueeze(1))

            _, next_word_idx = output.max(dim=1)
            inputs = next_word_idx.unsqueeze(1)
            hidden = as_variable(hidden.data)

        log.debug(
            "Decoder forward outputs: length = %d, size = %s",
            len(outputs),
            outputs[0].size(), )

        outputs = torch.cat(outputs, dim=1)
        log.debug("Decoder forward output: %s", outputs.size())

        return outputs, hidden
