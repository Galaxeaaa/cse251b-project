import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os, os.path
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random

import utils


class EncoderDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EncoderDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.LSTMCell(input_size, hidden_size)
        self.decoder = nn.LSTMCell(output_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, predict_len=30):
        device = input_seq.device
        (seq_len, batch_size, fea_len) = input_seq.shape
        h, c = torch.zeros(batch_size, self.hidden_size, device=device), torch.zeros(
            batch_size, self.hidden_size, device=device
        )
        outputs = []

        for i in range(seq_len):
            h, c = self.encoder(input_seq[i], (h, c))

        # for i in range(predict_len):
        #     output = self.linear(h)
        #     outputs.append(output)
        #     h, c = self.encoder(output, (h, c))

        for i in range(predict_len):
            output = self.linear(h)
            outputs.append(output)
            h, c = self.decoder(output, (h, c))

        outputs = torch.stack(outputs, dim=0)
        # outputs = outputs.permute(1, 0, 2)

        return outputs


def toModelFormat(tensor, masks):
    # Remove agents that are not in the scene
    valid_indices = torch.nonzero(masks.ravel()).squeeze()
    # Reshape input and target to [batch_size, seq_len, feature_size], the original batch_size and agent_size are combined
    tensor = tensor.reshape(-1, tensor.size(2), tensor.size(3))[valid_indices]
    tensor = tensor.permute(1, 0, 2)  # [seq_len, batch_size, feature_size]
    tensor = tensor.float()
    tensor = tensor[..., :2]

    return tensor


def normalize(data_fr_loader, mean_p=None, min_p=None, max_p=None):
    norm = data_fr_loader.clone().float()  # [batch_size, agent_size, seq_len, feature_size]
    # p = norm[..., :2]
    # v = norm[..., 2:]
    if mean_p is None:
        # mean_p = torch.mean(p, axis=0)
        # mean_p = torch.mean(mean_p, axis=0)
        mean_p = norm[..., 0, :2].unsqueeze(2).clone()
    norm[..., :2] = norm[..., :2] - mean_p
    # if min_p is None or max_p is None:
    #     max_p = torch.max(p, axis=0)[0]
    #     max_p = torch.max(max_p, axis=0)[0]
    #     min_p = torch.min(p, axis=0)[0]
    #     min_p = torch.min(min_p, axis=0)[0]
    # scale = (max_p - min_p) / 20
    # pv_norm[..., :2] = (p - min_p) / scale - 10  # normalize p_in to [-1, 1]
    # pv_norm[..., 2:] = v / scale  # scale v_in
    return norm, mean_p, min_p, max_p


def denormalize(norm, mean_p, min_p, max_p):
    return norm + mean_p


def train(
    model,
    dataloader,
    criterion,
    optimizer,
    num_epochs=10,
    vis_every=10,
    print_every=10,
):
    model.train()
    all_losses = []
    print("Start Training")
    total_loss = 0
    i = 0
    inp_vis, tar_vis = None, None
    for epoch in range(num_epochs):
        progress_bar = tqdm.tqdm(dataloader, ncols=100)
        for inp, tar, masks in progress_bar:
            if vis_every:
                if inp_vis is None:
                    inp_vis = inp
                if tar_vis is None:
                    tar_vis = tar

            inp_norm, mean_inp, min_inp, max_inp = normalize(inp)
            # tar_norm, _, _, _ = normalize(tar, mean_inp, min_inp, max_inp)
            inp_norm = toModelFormat(inp_norm, masks)
            # tar_norm = toModelFormat(tar_norm, masks)
            mean_inp = toModelFormat(mean_inp, masks)

            optimizer.zero_grad()
            out_norm = model(inp_norm.to(device))
            out = denormalize(out_norm, mean_inp.to(device), min_inp, max_inp)
            tar = toModelFormat(tar, masks)
            loss = criterion(out.to(device), tar.to(device))
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
        if vis_every and (epoch + 1) % vis_every == 0:
            visualize(model, inp_vis, tar_vis, masks)
        model.train()

    return all_losses


def evaluate(model, dataloader, criterion):
    model.eval()
    current_loss = 0
    with torch.no_grad():
        for i, (inp, tar, masks) in enumerate(dataloader):
            inp_norm, mean_inp, min_inp, max_inp = normalize(inp)
            # tar_norm, _, _, _ = normalize(tar, mean_inp, min_inp, max_inp)
            inp_norm = toModelFormat(inp_norm, masks)
            # tar_norm = toModelFormat(tar_norm, masks)
            mean_inp = toModelFormat(mean_inp, masks)

            out_norm = model(inp_norm.to(device))
            out = denormalize(out_norm, mean_inp.to(device), min_inp, max_inp)
            tar = toModelFormat(tar, masks)
            loss = criterion(out.to(device), tar.to(device))
            current_loss += loss.item()

    return current_loss / len(dataloader)


def visualize(model, inp, tar, masks):
    # inp, tar: [batch_size, seq_len, feature_size]
    model.eval()
    with torch.no_grad():
        inp_norm, mean_inp, min_inp, max_inp = normalize(inp)
        # tar_norm, _, _, _ = normalize(tar, mean_inp, min_inp, max_inp)
        inp_norm = toModelFormat(inp_norm, masks)
        # tar_norm = toModelFormat(tar_norm, masks)
        mean_inp = toModelFormat(mean_inp, masks)

        out_norm = model(inp_norm.to(device))
        out = denormalize(out_norm, mean_inp.to(device), min_inp, max_inp)
        inp = toModelFormat(inp, masks)
        tar = toModelFormat(tar, masks)
        loss = criterion(out.to(device), tar.to(device))

        # # Inverse normalize the output
        # out = out_norm.cpu()
        # out = out.reshape(out.size(0), out.size(1), 60, 4).permute(1, 2, 0, 3)
        # scale = (max_inp - min_inp) / 20
        # out[..., :2] = (out[..., :2] + 10) * scale + min_inp + mean_inp
        # out[..., 2:] = out[..., 2:] * scale

        # # Plot the output
        # inp = inp.squeeze(0).numpy()  # [agent_size=60, seq_len=19, feature_size=4]
        # tar = tar.squeeze(0).numpy()  # [agent_size=60, seq_len=30, feature_size=4]
        # out = out.squeeze(0).numpy()  # [agent_size=60, seq_len=30, feature_size=4]
        # plt.scatter(inp[0, :, 0], inp[0, :, 1], c="b", s=2)
        # plt.scatter(tar[0, :, 0], tar[0, :, 1], c="g", s=2)
        # plt.scatter(out[0, :, 0], out[0, :, 1], c="r", s=2)
        # plt.show()

        # Plot inp_norm and tar_norm in another figure
        inp = inp.cpu()
        tar = tar.cpu()
        out = out.cpu()
        plt.scatter(inp[:, 0, 0], inp[:, 0, 1], c="b", s=2)
        plt.scatter(tar[:, 0, 0], tar[:, 0, 1], c="g", s=2)
        plt.scatter(out[:, 0, 0], out[:, 0, 1], c="r", s=2)
        plt.show()

        return loss


def my_collate(batch):
    """collate lists of samples into batches, create [ batch_sz x agent_sz x seq_len x feature]"""
    inp = [np.dstack([scene["p_in"], scene["v_in"]]) for scene in batch]
    inp = np.array(inp)
    out = [np.dstack([scene["p_out"], scene["v_out"]]) for scene in batch]
    out = np.array(out)
    inp = torch.LongTensor(inp)
    out = torch.LongTensor(out)
    masks = [scene["car_mask"] for scene in batch]
    masks = np.array(masks)
    masks = torch.tensor(masks).squeeze()
    return [inp, out, masks]


def transform_batch(batch):
    # Remove agents that are not in the scene
    valid_indices = torch.nonzero(masks.ravel()).squeeze()
    inp = inp.reshape(-1, inp.size(2), inp.size(3))[valid_indices]
    tar = tar.reshape(-1, tar.size(2), tar.size(3))[valid_indices]
    if vis_every:
        if inp_vis is None:
            inp_vis = inp[0:1]
        if tar_vis is None:
            tar_vis = tar[0:1]
    # Process input and target
    inp_norm, mean_inp, min_inp, max_inp = normalize(inp)
    # tar_norm, _, _, _ = normalize(tar, mean_inp, min_inp, max_inp)
    inp_norm = (
        inp_norm.permute(1, 0, 2).float().to(device)
    )  # [seq_len, batch_size, feature_size]
    tar = tar.permute(1, 0, 2).float().to(device)  # [seq_len, batch_size, feature_size]


if __name__ == "__main__":
    # Configure
    data_path = "./data/"
    city_index_path = "./"
    model_path = "./model/"
    model_file = "encoder_decoder_0.pt"
    model_path = ""
    batch_size = 16
    cutoff = None

    do_train = True
    input_size = 2
    hidden_size = 200
    output_size = 2
    lr = 0.001
    n_epochs = 5
    vis_every = 0

    do_save = True
    save_path = "./model/"
    save_file = "encoder_decoder_5.pt"
    # save_file = "seq2seq_attention.pt"

    do_evaluate = True

    do_visualize = True
    n_vis = 10

    do_output = True
    submission_path = "./output/"
    submission_file = "./EncoderDecoder_MIA.csv"

    # Load data
    (
        MIA_train_loader,
        PIT_train_loader,
        MIA_valid_loader,
        PIT_valid_loader,
        MIA_train_dataset,
        PIT_train_dataset,
        MIA_valid_dataset,
        PIT_valid_dataset,
    ) = utils.loadData(
        data_path,
        city_index_path,
        batch_size=batch_size,
        collate_fn=my_collate,
        cutoff=cutoff,
    )

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    if model_path and model_file and os.path.exists(model_path):
        print(f"Loading model from {model_path}...", end="")
        model = EncoderDecoder(input_size, hidden_size, output_size).to(device)
        model.load_state_dict(torch.load(model_path + model_file))
        print("Done")
    else:
        model = EncoderDecoder(input_size, hidden_size, output_size).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train model
    if do_train:
        MIA_train_losses = train(
            model,
            MIA_train_loader,
            criterion,
            optimizer,
            num_epochs=n_epochs,
            vis_every=vis_every,
        )
        plt.plot(MIA_train_losses)
        plt.show()

    # Save model
    if do_save:
        print(f"Saving model to {save_path}... ", end="")
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        torch.save(model.state_dict(), save_path + save_file)
        print("Done")

    # Evaluate model
    if do_evaluate:
        print("Evaluating...", end="")
        MIA_valid_loss = evaluate(model, MIA_valid_loader, criterion)
        print(f"MIA validation Loss: {MIA_valid_loss:.4f}")
        print("Done")

    # Visualize model
    if do_visualize:
        print("Visualizing...", end="")
        for i in range(n_vis):
            inp, tar, masks = next(iter(MIA_valid_loader))
            loss = visualize(model, inp, tar, masks)
        print("Done")

    if do_output:
        print("Generating submission...", end="")
        # Load test data

