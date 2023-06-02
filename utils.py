import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os, os.path
import numpy
import pickle
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
        pkl_path = pkl_path.replace("./", "").replace("\\", "/")

        with open(os.path.join(self.path_data, pkl_path), "rb") as f:
            data = pickle.load(f)

        if self.transform:
            data = self.transform(data)

        return data


def default_collate(batch):
    """collate lists of samples into batches, create [ batch_sz x agent_sz x seq_len x feature]"""
    inp = [numpy.dstack([scene["p_in"], scene["v_in"]]) for scene in batch]
    inp = numpy.array(inp)
    out = [numpy.dstack([scene["p_out"], scene["v_out"]]) for scene in batch]
    out = numpy.array(out)
    inp = torch.LongTensor(inp)
    out = torch.LongTensor(out)
    return [inp, out]


def collate_with_len(batch):
    inp = [numpy.dstack([scene["p_in"], scene["v_in"]]) for scene in batch]
    inp = numpy.array(inp)
    out = [numpy.dstack([scene["p_out"], scene["v_out"]]) for scene in batch]
    out = numpy.array(out)
    inp = torch.LongTensor(inp)
    out = torch.LongTensor(out)
    mask = [scene["car_mask"] for scene in batch]
    mask = torch.tensor(mask).squeeze()
    return [inp, out, mask]

def collate_with_lane(batch):
    inp = [numpy.dstack([scene["p_in"], scene["v_in"]]) for scene in batch]
    inp = numpy.array(inp)
    out = [numpy.dstack([scene["p_out"], scene["v_out"]]) for scene in batch]
    out = numpy.array(out)
    inp = torch.LongTensor(inp)
    out = torch.LongTensor(out)
    mask = [scene["car_mask"] for scene in batch]
    mask = torch.tensor(mask).squeeze()
    lane = [torch.tensor(scene["lane"]).float() for scene in batch]
    # lane = torch.tensor(lane).squeeze()
    lane_norm = [torch.tensor(scene["lane_norm"]).float() for scene in batch]
    # lane_norm = torch.tensor(lane_norm).squeeze()
    return [inp, out, mask,lane,lane_norm]


def loadData(
    path, city_index_path, batch_size=4, split=0.9, cutoff=None, collate_fn=default_collate
):
    # split train and valid data
    # load at most cutoff sample (only when get city data)

    if collate_fn is None:
        collate_fn = default_collate

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
        collate_fn=collate_fn,
        num_workers=0,
    )
    PIT_train_loader = DataLoader(
        PIT_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    MIA_valid_loader = DataLoader(
        MIA_valid_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    PIT_valid_loader = DataLoader(
        PIT_valid_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
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


def visualization(sample,pred_X,pred_Y,traj_idx):

    plt.figure(figsize=(16, 16))

    p_in,p_out = sample['p_in'],sample['p_out']

    px,py = p_in[traj_idx,:,0],p_in[traj_idx,:,1]
    outx,outy = p_out[traj_idx,:,0],p_out[traj_idx,:,1]

    for i in range(len(sample["lane"])):
        x0,y0 = sample["lane"][i]
        vx,vy = sample["lane_norm"][i]
        
        plt.plot([x0-vx/2,x0+vx/2],[y0-vy/2,y0+vy/2])

    plt.scatter(px,py,label = "Input",s = 10)

    plt.scatter(pred_X,pred_Y,label = "Predict",s = 10)

    plt.scatter(outx,outy,label = "Groudtruth",s = 10)

    plt.legend()

    plt.show()

def loadValidData_by_traj(path):
    print("Load valid data in traj level")
    pkl_list = glob(os.path.join(path, "*"))
    inps = []
    scene_ids = []
    # print(len(pkl_list))
    for pkl_path in pkl_list:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
            agent_id = data["agent_id"]
            track_id = data["track_id"]
            # print(type(agent_id),type(track_id),agent_id,track_id)
            indices = np.where(track_id == agent_id)[0]
            # print(indices)
            inp = numpy.dstack([data["p_in"], data["v_in"]])
            inp = numpy.array(inp)[indices]
            # print(inp.shape)
            inps.append(inp)
            scene_ids.append(data["scene_idx"])
    inps = torch.tensor(inps).squeeze()
    return scene_ids,inps

def formOutput(path,data,scene_ids,name):
    output = data.reshape(data.shape[0],-1).to("cpu")
    df = pd.DataFrame(output.detach().numpy())
    df.columns = ["v"+str(i+1) for i in range(60)]
    df.insert(0, 'ID', scene_ids)
    df.to_csv(path+name, index=False)

def get_nearest_lane(points,lane,lane_norm):
    ''' a * [x,y] / b * [x1,y1,x2,y2] '''
    x = points[:, 0].unsqueeze(1)
    y = points[:, 1].unsqueeze(1)
    x1 = (lane[:, 0] - lane_norm[:,0]/2).unsqueeze(0)
    y1 = (lane[:, 1] - lane_norm[:,1]/2).unsqueeze(0)
    x2 = (lane[:, 0] + lane_norm[:,0]/2).unsqueeze(0)
    y2 = (lane[:, 1] + lane_norm[:,1]/2).unsqueeze(0)

    m = (y2 - y1) / (x2 - x1)
    c = y1 - m * x1

    s_distances = torch.abs(m * x - y + c) / torch.sqrt(m**2 + 1)
    l_distances = torch.sqrt((y-y1)**2+(x-x1)**2)
    r_distances = torch.sqrt((y-y2)**2+(x-x2)**2)

    distance = torch.cat([s_distances,l_distances,r_distances],dim=1)

    indices = torch.argmin(distance,dim = 1) % 20

    nearest_lane = lane[indices]
    nearest_lane_norm = lane_norm[indices]

    return nearest_lane,nearest_lane_norm

def merge_output(datapath,outpath,MIAname,PITname,mergeName):
    print("Load valid data in traj level")
    MIA_df = pd.read_csv(outpath+MIAname)
    PIT_df = pd.read_csv(outpath+PITname)
    merge_df = pd.DataFrame(columns=MIA_df.columns)
    pkl_list = glob(os.path.join(datapath, "*"))
    idx = []
    # print(len(pkl_list),len(merge_df))
    for i,pkl_path in enumerate(pkl_list):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
            if data["city"] == "MIA":
                row_to_add = MIA_df.iloc[[i]]
            else:
                row_to_add = PIT_df.iloc[[i]]
            merge_df = pd.concat([merge_df, row_to_add], ignore_index=True)
    merge_df.to_csv(outpath+mergeName, index=False)

datapath = 'C:\\Users\\zxk\\Desktop\\251B\\class-proj\\ucsd-cse-251b-class-competition\\val_in\\val_in'
outpath = "C:\\Users\\zxk\\Desktop\\251B\\class-proj\\ucsd-cse-251b-class-competition\\"
MIAname = "LSTM3.csv"
PITname = "LSTM2.csv"
mergeName = "LTSM_M.csv"
# merge_output = merge_output(datapath,outpath,MIAname,PITname,mergeName)