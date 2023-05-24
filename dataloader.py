import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os, os.path
import numpy
import pickle
from glob import glob


class ADataset(Dataset):
    """Dataset class for Argoverse"""

    def __init__(self, path_data, path_list, left, right, transform=None):
        super(ADataset, self).__init__()

        self.path_data = path_data
        self.pkl_list = path_list
        self.l = left
        self.r = right
        self.transform = transform

    def __len__(self):
        return self.r - self.l

    def __getitem__(self, idx):
        pkl_path = self.pkl_list[self.l + idx]
        with open(self.path_data + pkl_path, "rb") as f:
            data = pickle.load(f)

        if self.transform:
            data = self.transform(data)

        return data


def my_collate(batch):
    """collate lists of samples into batches, create [ batch_sz x agent_sz x seq_len x feature]"""
    inp = [numpy.dstack([scene["p_in"], scene["v_in"]]) for scene in batch]
    inp = numpy.array(inp)
    out = [numpy.dstack([scene["p_out"], scene["v_out"]]) for scene in batch]
    out = numpy.array(out)
    inp = torch.LongTensor(inp)
    out = torch.LongTensor(out)
    return [inp, out]


def collate_normalize(batch):
    """collate lists of samples into batches, create [ batch_sz x agent_sz x seq_len x feature]"""
    """!!!UNFINISHED!!! DONT USE THIS FUNCTION!!!"""

    def normalize(pv):
        res = pv.copy()
        # normalize p_in and scale v_in in each scene
        for scene_i in range(len(pv)):
            mean_p = numpy.mean(res[scene_i][:, :, 0], axis=0)
            res[scene_i][:, :, 0] = pv[scene_i][:, :, 0] - mean_p
            max_p = numpy.max(res[scene_i][:, :, 0])
            min_p = numpy.min(res[scene_i][:, :, 0])
            scale = max_p - min_p / 2
            res[scene_i][:, :, 0] = (pv[scene_i][:, :, 0] - min_p) / scale - 1
            res[scene_i][:, :, 1] = pv[scene_i][:, :, 1] / scale

    inp = [numpy.dstack([scene["p_in"], scene["v_in"]]) for scene in batch]
    inp = numpy.array(inp)
    normalize(inp)
    out = [numpy.dstack([scene["p_out"], scene["v_out"]]) for scene in batch]
    out = numpy.array(out)
    normalize(out)
    inp = torch.LongTensor(inp)
    out = torch.LongTensor(out)
    return [inp, out]


def loadData(
    path, city_index_path, batch_size=4, split=0.9, cutoff=None, normalize=False
):
    # split train and valid data
    # load at most cutoff sample (only when get city data)

    if not os.path.exists(path):
        raise Exception("Wrong Path:" + path)

    mia_path = city_index_path + "MIA.pkl"
    pit_path = city_index_path + "PIT.pkl"

    if os.path.exists(mia_path) and os.path.exists(pit_path):
        print("Reading city index file...")
        with open(mia_path, "rb") as file:
            MIA_list = pickle.load(file)
            if cutoff is not None:
                MIA_list = MIA_list[: int(cutoff)]
        with open(pit_path, "rb") as file:
            PIT_list = pickle.load(file)
            if cutoff is not None:
                PIT_list = PIT_list[: int(cutoff)]
    else:
        print("Get City File")
        pkl_list = glob(os.path.join(path, "*"))[:cutoff]
        pkl_list.sort()
        MIA_list = []
        PIT_list = []
        for pkl_path in pkl_list:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
                if data["city"] == "MIA":
                    MIA_list.append(pkl_path)
                elif data["city"] == "PIT":
                    PIT_list.append(pkl_path)
                else:
                    raise Exception("Unknown City: " + data["city"])

        MIA_list.sort()
        PIT_list.sort()

        with open(mia_path, "wb") as file:
            pickle.dump(MIA_list, file)
        with open(mia_path, "wb") as file:
            pickle.dump(PIT_list, file)

    MIA_S = int(len(MIA_list) * split)
    PIT_S = int(len(PIT_list) * split)

    MIA_train_dataset = ADataset(path, MIA_list, 0, MIA_S)
    MIA_valid_dataset = ADataset(path, MIA_list, MIA_S, len(MIA_list))
    PIT_train_dataset = ADataset(path, PIT_list, 0, PIT_S)
    PIT_valid_dataset = ADataset(path, PIT_list, PIT_S, len(PIT_list))

    MIA_train_loader = DataLoader(
        MIA_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=my_collate,
        num_workers=0,
    )
    PIT_train_loader = DataLoader(
        PIT_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=my_collate,
        num_workers=0,
    )
    MIA_valid_loader = DataLoader(
        MIA_valid_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=my_collate,
        num_workers=0,
    )
    PIT_valid_loader = DataLoader(
        PIT_valid_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=my_collate,
        num_workers=0,
    )

    return (
        MIA_train_loader,
        PIT_train_loader,
        MIA_valid_loader,
        PIT_valid_loader,
        MIA_train_dataset,
        PIT_train_dataset,
        MIA_valid_dataset,
        PIT_valid_dataset,
    )
