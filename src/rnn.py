import torch
import torch.nn.functional as F
import torch.nn as nn
import logging
from tokens import SOS_TOKEN, PAD_TOKEN
import pdb

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
        self.decoder = AttnDecoderRNN(hidden_size, output_size, 'dot',
                                      decoder_layers)

    def forward(self, inputs, targets=None, encoder_hidden=None):
        if not encoder_hidden:
            encoder_hidden = self.init_hidden(inputs.size(0))

        encoder_outputs, encoder_hidden = self.encoder(inputs, encoder_hidden)

        decoder_hidden = torch.cat(encoder_hidden, 1).unsqueeze(0)
        log.debug("Decoder hidden [Seq2Seq]: %s", decoder_hidden.size())

        decoder_output, decoder_hidden = self.decoder(decoder_hidden,
                                                      encoder_outputs)
        return decoder_output, decoder_hidden

    def init_hidden(self, batch_size):
        return as_variable(torch.zeros(2, batch_size, self.hidden_size))


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=True)

    def forward(self, inputs, hidden):
        log.debug("\n\nEncoder inputs: %s", inputs.size())
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
        # B x S x C -> S x B x C
        inputs_sorted = inputs_sorted.transpose(0, 1)

        # NOTE: this will change S output dimension
        # to length of the longest sequence
        # packed = nn.utils.rnn.pack_padded_sequence(inputs_sorted, lengths)
        # packed_out, packed_hidden = self.gru(packed, hidden)

        # output, _ = nn.utils.rnn.pad_packed_sequence(packed_out)

        packed_out, packed_hidden = self.gru(embedded.transpose(0, 1), hidden)
        output = packed_out

        hidden = packed_hidden
        log.debug("Encoder hidden: %s", packed_hidden.size())
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


class AttentionLayer(nn.Module):
    def __init__(self, method, hidden_size):
        super(AttentionLayer, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)

        # Create variable to store attention energies
        # B x 1 x S
        attn_energies = as_variable(
            torch.zeros(batch_size, seq_len).unsqueeze(1))

        log.debug("AttentionLayer.attn_energies: %s",
                  attn_energies.data.size())

        # Calculate energies for each encoder output
        for i in range(seq_len):
            attn_energies[:, :, i] = self.score(hidden, encoder_outputs[i])

        # Normalize energies to weights in range 0 to 1,
        # resize to B x 1 x S
        return F.softmax(attn_energies)

    def score(self, hidden, encoder_output):
        """
        Returns attention score for a particular encoded token.
        Outputs: energy, a vector of size B x 1
        """
        log.debug("AttentionLayer.score[hidden]: %s", hidden.size())
        log.debug("AttentionLayer.score[encoder_output]: %s",
                  encoder_output.size())

        # pdb.set_trace()

        if self.method == 'dot':
            energy = torch.bmm(
                hidden.unsqueeze(1), encoder_output.unsqueeze(2))
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.other.dot(energy)
            return energy


class AttnDecoderRNN(nn.Module):
    def __init__(self,
                 hidden_size,
                 output_size,
                 attn_model,
                 n_layers=1,
                 dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()

        # Keep parameters for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(
            hidden_size * 2, hidden_size * 2, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size * 4, output_size)

        self.attn = AttentionLayer(attn_model, hidden_size)

    def _forward_step(self, word_input, last_context, last_hidden,
                      encoder_outputs):
        # Note: we run this one step at a time

        batch_size = encoder_outputs.size(1)

        # Get the embedding of the current input word (last output word)
        # 1 x B x N
        word_embedded = self.embedding(word_input).view(1, batch_size, -1)
        log.debug("AttnDecoderRNN._forward_step[word_embedded]: %s",
                  word_embedded.size())

        # Combine embedded input word and last context, run through RNN
        rnn_input = torch.cat((word_embedded, last_context.unsqueeze(0)), 2)
        log.debug(
            "AttnDecoderRNN._forward_step: rnn_input = %s, last_hidden = %s",
            rnn_input.size(), last_hidden.size())
        rnn_output, hidden = self.gru(rnn_input, last_hidden)
        log.debug("AttnDecoderRNN._forward_step: rnn_output = %s, hidden = %s",
                  rnn_output.size(), hidden.size())

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs
        attn_weights = self.attn(rnn_output.squeeze(0), encoder_outputs)
        log.debug("AttnDecoderRNN._forward_step[attn_weights]: %s",
                  attn_weights.size())

        pdb.set_trace()
        # B x 1 x N
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))

        # Final output layer (next word prediction)
        # using the RNN hidden state and context vector
        # 1 x B x N -> B x N
        rnn_output = rnn_output.squeeze(0)
        # B x S=1 x N -> B x N
        context = context.squeeze(1)
        log.debug("AttnDecoderRNN._forward_step[rnn_output]: %s",
                  rnn_output.size())

        log.debug("AttnDecoderRNN._forward_step[context]: %s", context.size())

        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)))

        return output, context, hidden, attn_weights

    def forward(self, hidden, encoder_outputs):

        log.debug("AttnDecoderRNN: encoder_outputs %s", encoder_outputs.size())

        batch_size = encoder_outputs.size(1)
        target_length = encoder_outputs.size(0)

        log.debug(
            "AttnDecoderRNN parameters: batch_size = %d, target_length = %d",
            batch_size, target_length)

        inputs = torch.LongTensor([SOS_TOKEN]).repeat(batch_size, 1)
        inputs = as_variable(inputs)
        attn_ctx = as_variable(
            torch.FloatTensor(batch_size, self.hidden_size).random_())

        outputs = []

        for step in range(target_length):
            output, attn_ctx, hidden, _ = self._forward_step(
                inputs,
                attn_ctx,
                hidden,
                encoder_outputs, )

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
