import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os, os.path
import tqdm
import matplotlib.pyplot as plt
import numpy as np

import utils


class EncoderDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EncoderDecoder, self).__init__()
        self.hidden_size = hidden_size

        self.encoder = nn.LSTMCell(input_size, hidden_size)
        # self.attention = nn.MultiheadAttention(hidden_size, 1)
        self.decoder = nn.LSTMCell(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(hidden_size, output_size)

        self.initWeights()

    def forward(self, inp):
        batch_size = inp.size(1)
        hidden_encoder = self.initHidden(batch_size).to(device)
        cell_encoder = self.initCell(batch_size).to(device)
        hidden_decoder = self.initHidden(batch_size).to(device)
        cell_decoder = self.initCell(batch_size).to(device)

        for i in range(19):
            hidden_encoder, cell_encoder = self.encoder(
                inp[i], (hidden_encoder, cell_encoder)
            )

        hidden_decoder = hidden_encoder
        output = [hidden_decoder]
        for i in range(29):
            hidden_decoder, cell_decoder = self.decoder(
                hidden_decoder, (hidden_decoder, cell_decoder)
            )
            output.append(hidden_decoder)

        output = torch.stack(output, dim=0)
        output = self.relu(output)
        output = self.linear(output)
        return output

    def initWeights(self):
        self.linear.weight.data.normal_(0.0, 0.02)

    def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

    def initCell(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size, hidden_size)

    def forward(self, input_seq):
        hidden = torch.zeros(1, input_seq.size(1), self.hidden_size).to(device)
        context_seq, hidden = self.gru(input_seq, hidden)
        return context_seq, hidden

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size

        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

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


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        self.gru = nn.GRUCell(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, context, hidden):
        hidden = self.gru(context, hidden)
        output = self.linear(hidden)
        return output, hidden


class Seq2SeqAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2SeqAttention, self).__init__()
        self.encoder = Encoder(input_size, hidden_size)
        self.attention = Attention(hidden_size)
        self.decoder = Decoder(hidden_size, output_size)

    def forward(self, input_seq, len_out=30):
        context_seq, hidden_encoder = self.encoder(input_seq)
        hidden_decoder = hidden_encoder.squeeze(0)

        # outputs = torch.zeros(output_seq_len, batch_size, output_size)
        outputs = []

        for t in range(len_out):
            context, _ = self.attention(hidden_decoder, context_seq)
            output, hidden_decoder = self.decoder(context, hidden_decoder)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=0)

        return outputs


def normalize(pv, mean_p=None, min_p=None, max_p=None):
    pv_norm = pv.clone()  # [batch_size, agent_size, seq_len, feature_size]
    pv_norm = pv_norm.type(torch.float32)
    # Normalize p_in and scale v_in in each scene
    p = pv_norm[..., :2]
    v = pv_norm[..., 2:]
    if mean_p is None:
        mean_p = torch.mean(p, axis=1, keepdim=True)
        mean_p = torch.mean(mean_p, axis=2, keepdim=True)
        # mean_p = p[:, :, 0, :].unsqueeze(2)
    pv_norm[..., :2] = p - mean_p
    if min_p is None or max_p is None:
        max_p = torch.max(p, axis=1, keepdim=True)[0]
        max_p = torch.max(max_p, axis=2, keepdim=True)[0]
        min_p = torch.min(p, axis=1, keepdim=True)[0]
        min_p = torch.min(min_p, axis=2, keepdim=True)[0]
    scale = max_p - min_p / 20
    pv_norm[..., :2] = (p - min_p) / scale - 10  # normalize p_in to [-1, 1]
    pv_norm[..., 2:] = v / scale  # scale v_in
    return pv_norm, mean_p, min_p, max_p


def train(
    model,
    dataloader,
    criterion,
    optimizer,
    num_epochs=10,
    print_every=10,
    plot_every=1,
):
    model.train()
    all_losses = []
    print("Start Training")
    total_loss = 0
    i = 0
    for epoch in range(num_epochs):
        progress_bar = tqdm.tqdm(dataloader, ncols=100)
        for inp, tar in progress_bar:
            # Process input and target
            inp, mean_inp, min_inp, max_inp = normalize(inp)
            tar, _, _, _ = normalize(tar, mean_inp, min_inp, max_inp)
            inp = (
                inp.permute(2, 0, 1, 3)
                .reshape(inp.size(2), inp.size(0), -1)
                .float()
                .to(device)
            )  # [seq_len, batch_size, feature_size]
            tar = (
                tar.permute(2, 0, 1, 3)
                .reshape(tar.size(2), tar.size(0), -1)
                .float()
                .to(device)
            )  # [seq_len, batch_size, feature_size]

            optimizer.zero_grad()
            output = model(inp)
            loss = criterion(output, tar)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            if i % print_every == 0:
                progress_bar.set_description(
                    "Epoch: %d, Loss: %.4f\t" % (epoch + 1, loss.item())
                )
            i += 1
        all_losses.append(total_loss / len(dataloader))
        total_loss = 0

    return all_losses


def evaluate(model, dataloader, criterion):
    model.eval()
    current_loss = 0
    with torch.no_grad():
        for i, (inp, tar) in enumerate(dataloader):
            inp, mean_inp, min_inp, max_inp = normalize(inp)
            tar, _, _, _ = normalize(tar, mean_inp, min_inp, max_inp)
            inp = (
                inp.permute(2, 0, 1, 3)
                .reshape(inp.size(2), inp.size(0), -1)
                .float()
                .to(device)
            )
            tar = (
                tar.permute(2, 0, 1, 3)
                .reshape(tar.size(2), tar.size(0), -1)
                .float()
                .to(device)
            )

            out = model(inp)
            loss = criterion(out, tar)
            current_loss += loss.item()
    return current_loss / len(dataloader)


def visualize(model, dataloader, n_plot):
    model.eval()
    with torch.no_grad():
        dataloader_iter = iter(dataloader)
        for i in range(n_plot):
            inp, tar = next(dataloader_iter)
            inp = inp[:1]
            tar = tar[:1]
            inp_norm, mean_inp, min_inp, max_inp = normalize(inp)
            tar_norm, _, _, _ = normalize(tar, mean_inp, min_inp, max_inp)
            inp_norm = (
                inp_norm.permute(2, 0, 1, 3)
                .reshape(inp.size(2), inp.size(0), -1)
                .float()
                .to(device)
            )
            tar_norm = (
                tar_norm.permute(2, 0, 1, 3)
                .reshape(tar.size(2), tar.size(0), -1)
                .float()
                .to(device)
            )

            # out_norm = model(inp_norm)  # [seq_len, batch_size, feature_size]
            # loss = criterion(out_norm, tar_norm)
            # print(f"Loss of the {i+1}-th visualization: {loss.item():.4f}")

            # # Inverse normalize the output
            # out = out_norm.cpu()
            # out = out.reshape(out.size(0), out.size(1), 60, 4).permute(1, 2, 0, 3)
            # scale = max_inp - min_inp / 2
            # out[..., :2] = (out[..., :2] + 1) * scale + min_inp + mean_inp
            # out[..., 2:] = out[..., 2:] * scale

            # Plot the output
            inp = inp.squeeze(0).numpy()  # [agent_size=60, seq_len=19, feature_size=4]
            # out = out.squeeze(0).numpy()  # [agent_size=60, seq_len=30, feature_size=4]
            tar = tar.squeeze(0).numpy()  # [agent_size=60, seq_len=30, feature_size=4]
            plt.scatter(inp[0, :, 0], inp[0, :, 1], c="b")
            # plt.scatter(out[0, :, 0], out[0, :, 1], c="g")
            plt.scatter(tar[0, :, 0], tar[0, :, 1], c="r")
            plt.show()

            # # Plot inp_norm and tar_norm in another figure
            # inp_norm = inp_norm.cpu()
            # tar_norm = tar_norm.cpu()
            # out_norm = out_norm.cpu()
            # inp_norm = inp_norm.reshape(
            #     inp_norm.size(0), inp_norm.size(1), 60, 4
            # ).squeeze(1)
            # tar_norm = tar_norm.reshape(
            #     tar_norm.size(0), tar_norm.size(1), 60, 4
            # ).squeeze(1)
            # out_norm = out_norm.reshape(
            #     out_norm.size(0), out_norm.size(1), 60, 4
            # ).squeeze(1)
            # plt.scatter(inp_norm[:, 0, 0], inp_norm[:, 0, 1], c="b", alpha=0.5)
            # plt.scatter(tar_norm[:, 0, 0], tar_norm[:, 0, 1], c="g", alpha=0.5)
            # plt.scatter(out_norm[:, 0, 0], out_norm[:, 0, 1], c="r", alpha=0.5)
            # plt.show()


data_path = "./data/"
city_index_path = "./"

(
    MIA_train_loader,
    PIT_train_loader,
    MIA_valid_loader,
    PIT_valid_loader,
    MIA_train_dataset,
    PIT_train_dataset,
    MIA_valid_dataset,
    PIT_valid_dataset,
) = utils.loadData(data_path, city_index_path, batch_size=16, cutoff=200)

# print(next(iter(MIA_valid_loader))[0].size())
# print(MIA_valid_loader[0].keys())

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model
model_path = "./model/"
model_file = "encoder_decoder_0.pt"
model_path = ""
if model_path and model_file and os.path.exists(model_path):
    print(f"Loading model from {model_path}...", end="")
    model = Seq2SeqAttention(input_size=240, hidden_size=200, output_size=240).to(device)
    model.load_state_dict(torch.load(model_path + model_file))
    print("Done")
else:
    model = Seq2SeqAttention(input_size=240, hidden_size=200, output_size=240).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train model
do_train = False
if do_train:
    MIA_train_losses = train(
        model, MIA_train_loader, criterion, optimizer, num_epochs=10
    )
    plt.plot(MIA_train_losses)
    plt.show()

# Save model
do_save = False
if do_save:
    save_path = "./model/"
    save_file = "encoder_decoder.pt"
    save_file = "seq2seq_attention.pt"

    print(f"Saving model to {save_path}... ", end="")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    torch.save(model.state_dict(), save_path + save_file)
    print("Done")

do_evaluate = False
if do_evaluate:
    # Evaluate model
    print("Evaluating...", end="")
    MIA_valid_loss = evaluate(model, MIA_valid_loader, criterion)
    print(f"MIA validation Loss: {MIA_valid_loss:.4f}")
    print("Done")

do_visualize = True
if do_visualize:
    # Visualize model
    print("Visualizing...", end="")
    visualize(model, MIA_train_loader, 3)
    print("Done")
