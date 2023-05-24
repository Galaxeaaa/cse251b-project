import dataloader
from sklearn.linear_model import LinearRegression as LR
import torch


# data_path = "C:\\Users\\zxk\\Desktop\\251B\\class-proj\\ucsd-cse-251b-class-competition\\train\\train"
# city_idx_path = "C:\\Users\\zxk\\Desktop\\251B\\class-proj\\ucsd-cse-251b-class-competition\\"
# batch_size = 4
# cutoff = 1000
# MIA_train_loader,PIT_train_loader,MIA_valid_loader,PIT_valid_loader,MIA_train_dataset,PIT_train_dataset,MIA_valid_dataset,PIT_valid_dataset = dataloader.loadData(data_path,city_idx_path,batch_size,split=0.9,cutoff=cutoff)

# lr1 = LR()

# for i_batch, sample_batch in enumerate(MIA_train_loader):
#     inp, out = sample_batch # [batch_size, track_sum, seq_len, features]
    