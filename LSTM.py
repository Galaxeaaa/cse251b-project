import dataloader
import torch
import torch.nn as nn
import random
import torch.optim as optim
import tqdm
import datetime

agent_id = 3

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm_cell = nn.LSTMCell(input_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_seq, predict_len):
        device = input_seq.device
        (batch_size,seq_len,fea_len) = input_seq.shape
        input_seq = input_seq.permute(1,0,2)
        h,c = torch.zeros(batch_size, self.hidden_dim, device=device),torch.zeros(batch_size, self.hidden_dim,device=device)
        outputs = []

        for i in range(seq_len):
            h, c = self.lstm_cell(input_seq[i], (h, c))

        for i in range(predict_len):
            output = self.linear(h)
            outputs.append(output)
            h, c = self.lstm_cell(output, (h, c))

        outputs = torch.stack(outputs, dim=0)
        outputs = outputs.permute(1,0,2)
        
        return outputs

data_path = "C:\\Users\\zxk\\Desktop\\251B\\class-proj\\ucsd-cse-251b-class-competition\\"
city_idx_path = "C:\\Users\\zxk\\Desktop\\251B\\class-proj\\cse251b-project\\"
model_path = "C:\\Users\\zxk\\Desktop\\251B\\class-proj\\model\\"
mode = "train"
batch_size = 4
cutoff = None
MIA_train_loader,PIT_train_loader,MIA_valid_loader,PIT_valid_loader,MIA_train_dataset,PIT_train_dataset,MIA_valid_dataset,PIT_valid_dataset = dataloader.loadData(data_path,city_idx_path,batch_size,split=0.9,cutoff=cutoff)

input_size = 4
hidden_size = 200
output_size = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

'''all agents as a sample'''
# input_size = 60 * 4
# hidden_size = 200
# output_size = 60 * 4

# learning_rate = 1E-2
# epochs = 10

# model = LSTM(input_dim=input_size,hidden_dim=hidden_size,output_dim=output_size)

# optimizer = optim.Adam(model.parameters(),lr = learning_rate)
# criterion = nn.MSELoss()

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('Using device:', device)
# model = model.to(device)
# losses = []

# for epoch in range(epochs):
#     eloss = []
#     for i_batch, sample_batch in enumerate(MIA_train_loader):
#         inp, out = sample_batch # [batch_size, track_sum, seq_len, features]
#         inp = inp.permute(0,2,1,3)
#         out = out.permute(0,2,1,3)
#         inp = inp.reshape(inp.shape[0],inp.shape[1],-1).float()
#         out = out.reshape(out.shape[0],out.shape[1],-1).float()
#         inp,out = inp.to(device),out.to(device)
#         predict_len = out.shape[1]

#         optimizer.zero_grad()
#         predict = model(inp,predict_len)
#         loss = torch.sqrt(criterion(out,predict))
#         loss.backward()
#         optimizer.step()
#         eloss.append(loss.item())
#     avgloss = sum(eloss)/len(eloss)
#     print("Epoch:",epoch,"Loss:",loss)
#     losses.append(avgloss)

'''an agent as a example'''

if mode == "train":
    learning_rate = 1E-4
    epochs = 50

    model = LSTM(input_dim=input_size,hidden_dim=hidden_size,output_dim=output_size)
    model.load_state_dict(torch.load(model_path+'2023-05-24_12-01-40_model_50.pth'))

    optimizer = optim.Adam(model.parameters(),lr = learning_rate)
    criterion = nn.MSELoss()

    model = model.to(device)
    model.train()
    losses = []

    for epoch in range(epochs):
        eloss = []
        for i_batch, sample_batch in enumerate(MIA_train_loader):
            inp, out = sample_batch # [batch_size, track_sum, seq_len, features]
            inp, out = inp.reshape(-1,inp.shape[2],inp.shape[3]).float(),out.reshape(-1,out.shape[2],out.shape[3]).float()
            inp,out = inp.to(device),out.to(device)
            predict_len = out.shape[1]
            first_col = inp[:, 0, :2].clone()
            broadcasted_first_col = first_col.unsqueeze(1).expand(-1, inp.shape[1], -1)
            inp[:, :, :2] -=  broadcasted_first_col

            optimizer.zero_grad()

            predict = model(inp,predict_len)
            broadcasted_first_col = first_col.unsqueeze(1).expand(-1, predict.shape[1], -1)
            predict[:, :, :2] += broadcasted_first_col
            # print(predict.shape,out.shape)
            loss = criterion(out,predict)
            loss.backward()
            optimizer.step()
            eloss.append(loss.item())
            # if i_batch % 10 == 9:
            #     print("Epoch: {} Batch: {} Loss {:.4f}".format(epoch,i_batch+1,loss))  
            # break
        avgloss = sum(eloss)/len(eloss)
        print("Epoch:",epoch+1,"Loss:",loss)
        losses.append(avgloss)
        
        current_datetime = datetime.datetime.now()
        current_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), model_path+str(current_datetime)+'_model_'+str(epoch+1)+'.pth')
        # break
    # print(predict,out)

if mode == "test":
    model = LSTM(input_dim=input_size,hidden_dim=hidden_size,output_dim=output_size)
    model.load_state_dict(torch.load(model_path+'2023-05-24_12-25-32_model_50.pth'))

    model = model.to(device)

    model.eval()

    criterion = nn.MSELoss()

    tlosses = []

    for i_batch, sample_batch in enumerate(MIA_valid_loader):
        inp, out = sample_batch # [batch_size, track_sum, seq_len, features]
        inp, out = inp.reshape(-1,inp.shape[2],inp.shape[3]).float(),out.reshape(-1,out.shape[2],out.shape[3]).float()
        inp,out = inp.to(device),out.to(device)
        predict_len = out.shape[1]

        first_col = inp[:, 0, :2].clone()
        broadcasted_first_col = first_col.unsqueeze(1).expand(-1, inp.shape[1], -1)
        inp[:, :, :2] -=  broadcasted_first_col

        predict = model(inp,predict_len)

        broadcasted_first_col = first_col.unsqueeze(1).expand(-1, predict.shape[1], -1)
        predict[:, :, :2] += broadcasted_first_col

        out = out[:,:,:2]
        predict = predict[:,:,:2]
        loss = criterion(out,predict)
        tlosses.append(loss.item())

    # print(predict,out)

    print("Average MSE Loss: ",sum(tlosses)/len(tlosses))

    
