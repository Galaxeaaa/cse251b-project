import torch
import torch.nn as nn
import os, os.path

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size)

    def forward(self, input_seq):
        hidden = (
            torch.zeros(1, input_seq.size(1), self.hidden_size),
            torch.zeros(1, input_seq.size(1), self.hidden_size),
        )
        context_seq, hidden = self.lstm(input_seq, hidden)
        return context_seq, hidden


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size

        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

        self.initWeights()

    def forward(self, hidden_decoder, context_seq):
        hidden_decoder = hidden_decoder.repeat(
            context_seq.size(0), 1, 1
        )  # [seq_len, batch_size, hidden_size]

        energy = torch.tanh(self.attn(torch.cat((hidden_decoder, context_seq), dim=2)))
        attention_scores = torch.matmul(energy, self.v)
        attention_weights = torch.softmax(attention_scores, dim=0)
        attention_weights = attention_weights.unsqueeze(2)
        context = torch.sum(context_seq * attention_weights, dim=0)
        return context, attention_weights

    def initWeights(self):
        self.attn.weight.data.normal_(0.0, 0.02)


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        self.lstm_cell = nn.LSTMCell(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, context, hidden, cell):
        hidden, cell = self.lstm_cell(context, (hidden, cell))
        output = self.linear(hidden)
        return output, hidden, cell

    def initWeights(self):
        self.linear.weight.data.normal_(0.0, 0.02)


class Seq2SeqAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2SeqAttention, self).__init__()
        self.encoder = Encoder(input_size, hidden_size)
        self.attention = Attention(hidden_size)
        self.decoder = Decoder(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(hidden_size, output_size)

        self.linear.weight.data.normal_(0.0, 0.02)

    def forward(self, input_seq, len_out=30):
        context_seq, hidden_encoder = self.encoder(input_seq)
        context_seq = self.dropout(context_seq)
        hidden_decoder = hidden_encoder[0][0]
        cell_decoder = torch.zeros_like(hidden_decoder)

        # outputs = torch.zeros(output_seq_len, batch_size, output_size)
        outputs = []

        for t in range(len_out):
            context, _ = self.attention(hidden_decoder, context_seq)
            output, hidden_decoder, cell_decoder = self.decoder(
                context, hidden_decoder, cell_decoder
            )
            output = self.relu(output)
            output = self.linear(output)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=0)

        return outputs