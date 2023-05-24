import dataloader
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn


data_path = "C:\\Users\\zxk\\Desktop\\251B\\class-proj\\ucsd-cse-251b-class-competition\\"
city_idx_path = "C:\\Users\\zxk\\Desktop\\251B\\class-proj\\cse251b-project\\"
batch_size = 4
cutoff = 1000
MIA_train_loader,PIT_train_loader,MIA_valid_loader,PIT_valid_loader,MIA_train_dataset,PIT_train_dataset,MIA_valid_dataset,PIT_valid_dataset = dataloader.loadData(data_path,city_idx_path,batch_size,split=0.9,cutoff=cutoff)

X1 = torch.tensor([i for i in range(19)]).reshape(-1,1)
X2 = torch.tensor([i for i in range(19,49)]).reshape(-1,1)

tlosses = []

criterion = nn.MSELoss()

for i_batch, sample_batch in enumerate(MIA_valid_loader):

    inp, out = sample_batch
    inp, out = inp.reshape(-1,inp.shape[2],inp.shape[3]).float(),out.reshape(-1,out.shape[2],out.shape[3]).float()
    for j in range(inp.shape[0]):
        lrx = LinearRegression()
        lry = LinearRegression()
        px,py = inp[j,:,0],inp[j,:,1]
        lrx.fit(X1,px)
        lry.fit(X1,py)
        px_pred = lrx.predict(X2)
        py_pred = lry.predict(X2)
        # print(px_pred,py_pred)
        pred = torch.tensor([px_pred,py_pred]).T
        # print(pred.shape)
        # break
        loss = criterion(pred,out[j,:,:2])
        tlosses.append(loss.item())
    # break

print("Average Loss: ",sum(tlosses)/len(tlosses))
    