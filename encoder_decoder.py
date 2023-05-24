import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os, os.path
import tqdm
import matplotlib.pyplot as plt

import utils


class EncoderDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(EncoderDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder = nn.LSTMCell(input_size, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=1)
        self.decoder = nn.LSTMCell(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(
        self, input, hidden_encoder, hidden_decoder, cell_encoder, cell_decoder
    ):
        for i in range(input.size(0)):
            hidden_encoder, cell_encoder = self.encoder(
                input[i], (hidden_encoder, cell_encoder)
            )
        hidden_decoder = hidden_encoder
        output = [hidden_decoder]
        for i in range(29):
            hidden_decoder, cell_decoder = self.decoder(
                hidden_decoder, (hidden_decoder, cell_decoder)
            )
            output.append(hidden_decoder)

        output = torch.stack(output, dim=0)
        # output = self.dropout(output)
        output = self.linear(output)
        return output, hidden_encoder, hidden_decoder, cell_encoder, cell_decoder

    def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

    def initCell(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)


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
    progress_bar = tqdm.tqdm(range(num_epochs), ncols=100)
    total_loss = 0
    i = 0
    for epoch in progress_bar:
        for input, target in dataloader:
            # print(input.size(), target.size())
            input = (
                input.permute(2, 0, 1, 3)
                .reshape(input.size(2), input.size(0), -1)
                .float()
                .to(device)
            )
            target = (
                target.permute(2, 0, 1, 3)
                .reshape(target.size(2), target.size(0), -1)
                .float()
                .to(device)
            )
            batch_size = input.size(1)
            hidden_encoder = model.initHidden(batch_size).to(device)
            cell_encoder = model.initCell(batch_size).to(device)
            hidden_decoder = model.initHidden(batch_size).to(device)
            cell_decoder = model.initCell(batch_size).to(device)
            optimizer.zero_grad()
            output, hidden_encoder, hidden_decoder, cell_encoder, cell_decoder = model(
                input, hidden_encoder, hidden_decoder, cell_encoder, cell_decoder
            )
            loss = criterion(output, target)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            if i % print_every == 0:
                progress_bar.set_description(
                    "Epoch: %d, Loss: %.4f\t" % (epoch + 1, loss.item())
                )
            i += 1
        all_losses.append(loss.item())

    return all_losses


def evaluate(model, dataloader, criterion):
    model.eval()
    current_loss = 0
    with torch.no_grad():
        for i, (input, target) in enumerate(dataloader):
            input = (
                input.permute(2, 0, 1, 3)
                .reshape(input.size(2), input.size(0), -1)
                .float()
                .to(device)
            )
            target = (
                target.permute(2, 0, 1, 3)
                .reshape(target.size(2), target.size(0), -1)
                .float()
                .to(device)
            )
            batch_size = input.size(1)
            hidden_encoder = model.initHidden(batch_size).to(device)
            cell_encoder = model.initCell(batch_size).to(device)
            hidden_decoder = model.initHidden(batch_size).to(device)
            cell_decoder = model.initCell(batch_size).to(device)

            output, hidden_encoder, hidden_decoder, cell_encoder, cell_decoder = model(
                input, hidden_encoder, hidden_decoder, cell_encoder, cell_decoder
            )
            loss = criterion(output, target)
            current_loss += loss.item()
    return current_loss / len(dataloader)


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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_path = "./model/encoder_decoder.pt"
model_path = ""
if model_path and os.path.exists(model_path):
    print(f"Loading model from {model_path}...")
    model = EncoderDecoder(input_size=240, hidden_size=200, output_size=240).to(device)
    model.load_state_dict(torch.load(model_path))
else:
    model = EncoderDecoder(input_size=240, hidden_size=200, output_size=240).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

MIA_train_losses = train(model, MIA_train_loader, criterion, optimizer, num_epochs=1000)
plt.plot(MIA_train_losses)
plt.show()

save_path = "./model/encoder_decoder.pt"
torch.save(model.state_dict(), save_path)

MIA_valid_loss = evaluate(model, MIA_valid_loader, criterion)
print(f"MIA validation Loss: {MIA_valid_loss:.4f}")
