import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os, os.path
import numpy
import pickle
from glob import glob


class ADataset(Dataset):
    """Dataset class for Argoverse"""

    def __init__(self, path_list, left, right, transform=None):
        super(ADataset, self).__init__()

        self.pkl_list = path_list
        self.l = left
        self.r = right
        self.transform = transform

    def __len__(self):
        return self.r - self.l

    def __getitem__(self, idx):
        pkl_path = self.pkl_list[self.l + idx]
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        if self.transform:
            data = self.transform(data)

        return data


def my_collate(batch):
    """collate lists of samples into batches, create [ batch_sz x agent_sz x seq_len x feature]"""
    inp = [numpy.dstack([scene["p_in"], scene["v_in"]]) for scene in batch]
    out = [numpy.dstack([scene["p_out"], scene["v_out"]]) for scene in batch]
    inp = torch.LongTensor(inp)
    out = torch.LongTensor(out)
    return [inp, out]


def loadData(path, city_index_path, split=0.9, cutoff=None):
    # split train and valid data
    # load st most cutoff sample (only when get city data)

    if not os.path.exists(path):
        raise Exception("Wrong Path:" + path)

    if os.path.exists(city_index_path + "MIA.pkl") and os.path.exists(
        city_index_path + "PIT.pkl"
    ):
        print("Read City File")
        with open(city_index_path + "MIA.pkl", "rb") as file:
            MIA_list = pickle.load(file)
        with open(city_index_path + "PIT.pkl", "rb") as file:
            PIT_list = pickle.load(file)
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

        with open(city_index_path + "MIA.pkl", "wb") as file:
            pickle.dump(MIA_list, file)
        with open(city_index_path + "PIT.pkl", "wb") as file:
            pickle.dump(PIT_list, file)

    MIA_S = int(len(MIA_list) * split)
    PIT_S = int(len(PIT_list) * split)

    MIA_train_dataset = ADataset(MIA_list, 0, MIA_S)
    MIA_valid_dataset = ADataset(MIA_list, MIA_S, len(MIA_list))
    PIT_train_dataset = ADataset(PIT_list, 0, PIT_S)
    PIT_valid_dataset = ADataset(PIT_list, PIT_S, len(PIT_list))

    batch_sz = 4

    MIA_train_loader = DataLoader(
        MIA_train_dataset,
        batch_size=batch_sz,
        shuffle=False,
        collate_fn=my_collate,
        num_workers=0,
    )
    PIT_train_loader = DataLoader(
        PIT_train_dataset,
        batch_size=batch_sz,
        shuffle=False,
        collate_fn=my_collate,
        num_workers=0,
    )
    MIA_valid_loader = DataLoader(
        MIA_valid_dataset,
        batch_size=batch_sz,
        shuffle=False,
        collate_fn=my_collate,
        num_workers=0,
    )
    PIT_valid_loader = DataLoader(
        PIT_valid_dataset,
        batch_size=batch_sz,
        shuffle=False,
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


# MIA_train_loader,PIT_train_loader,MIA_valid_loader,PIT_valid_loader,MIA_train_dataset,PIT_train_dataset,MIA_valid_dataset,PIT_valid_dataset = dataloader.loadData("C:\\Users\\zxk\\Desktop\\251B\\class-proj\\ucsd-cse-251b-class-competition\\train\\train","C:\\Users\\zxk\\Desktop\\251B\\class-proj\\ucsd-cse-251b-class-competition\\",split=0.9,cutoff=None)
