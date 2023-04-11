import dataset 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal import StaticGraphTemporalSignal
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.nn.recurrent import GConvGRU
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import f1_score


num_nodes = 489


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, filters):
        super(RecurrentGCN, self).__init__()
        torch.manual_seed(1234567)
        self.encode_linear=nn.Linear(314,4)
        self.recurrent = GConvGRU(10, filters, 2)
        self.linear = torch.nn.Linear(filters, 1)

    def forward(self, x, edge_index):
        # x = num_nodes * 630 (original features dim)
        v0 = x[:,0:314]
        v1 = x[:,315:629]
        v0 = self.encode_linear(v0)
        v1 = self.encode_linear(v1)
        a = torch.cat([v0,x[:,314:315],v1,x[:,629:630]],dim=1)
        h = self.recurrent(a, edge_index)
        h = F.relu(h)
        h = self.linear(h)
        h = torch.sigmoid(h)
        return h


class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
 
    def forward(self, predict, target):
        pt = predict
        loss = - ((1 - self.alpha) * ((1 - pt+1e-5) ** self.gamma) * (target * torch.log(pt+1e-5)) +  self.alpha * (
                (pt++1e-5) ** self.gamma) * ((1 - target) * torch.log(1 - pt+1e-5)))
 
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


# this is a list of PyG `Data` objects
data_list=dataset.MyOwnDataset('dataset')

edge_index=data_list[0].edge_index
features=[]
targets=[]

for data in data_list:
    features.append(data.x)
    targets.append(data.y)

# this is the PyG Temporal iterator
dataset = StaticGraphTemporalSignal(
    edge_index=edge_index,
    edge_weight=None,
    features=features,
    targets=targets
)

train_dataset,test_dataset=temporal_signal_split(dataset,train_ratio=0.5)

# criterion = nn.BCELoss()
criterion = BCEFocalLoss(alpha=0.25, gamma=2)

model = RecurrentGCN(node_features=630, filters=32)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
print('Training device: ',device)

# train_dataset = train_dataset.to(device)
# test_dataset = test_dataset.to(device)

model = model.to(device)
criterion = criterion.to(device)

train_dataset_snapshots = [s.to(device) for s in train_dataset]
test_dataset_snapshots = [s.to(device) for s in test_dataset]
zero_tensor=torch.zeros(1).to(device)
none_tensor=torch.tensor(0).to(device)


x_index=[]
y_loss=[]
y_f1=[]


def round_tensor(t, decimal_places=3):
    return round(t.item(), decimal_places)


model.train()

for epoch in tqdm(range(50)):
    loss = 0
    f1 = 0
    for time, snapshot in enumerate(train_dataset_snapshots):
        y_hat = model(snapshot.x, snapshot.edge_index)
        loss += criterion(y_hat,snapshot.y.float())
        pred = y_hat.ge(.5).view(-1)
        # # sub-sampling here
        # y_true=snapshot.y # copy snapshot! do not modify it!
        # for i in range(len(snapshot.y)-1,-1,-1):
        #     if snapshot.y[i]==zero_tensor and random.random()>0.1:
        #         y_true=torch.cat([y_true[:i,:],y_true[i+1:,:]],dim=0)
        #         y_hat=torch.cat([y_hat[:i,:],y_hat[i+1:,:]],dim=0)
        # # print('dim_subsample: ',len(y_true))
        # loss += criterion(y_hat,y_true.float())
        # # end sub-sampling

        f1 += f1_score(pred.cpu().numpy(),snapshot.y.cpu().numpy(),zero_division=0)
    loss = loss / (time+1)
    f1 = f1 / (time+1)
    y_loss.append(round_tensor(loss))
    y_f1.append(round_tensor(f1))
    x_index.append(epoch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

plt.subplot(1,2,1)
plt.plot(x_index,y_loss,color='r')
plt.xlabel('epoches')
plt.title('Loss')
plt.subplot(1,2,2)
plt.plot(x_index,y_f1,color='g')
plt.xlabel('epoches')
plt.title('F1')
# plt.show()
plt.savefig('train.PNG')

torch.save(model, './model/GConvGRU.pkl') 


# evaluate the model
model.eval()
loss = 0
f1 = 0
for time, snapshot in enumerate(test_dataset_snapshots):
    y_hat = model(snapshot.x, snapshot.edge_index)
    loss += criterion(y_hat,snapshot.y.float())
    pred = y_hat.ge(.5).view(-1)
    f1 += f1_score(pred.cpu().numpy(),snapshot.y.cpu().numpy(),zero_division=0)
loss = loss / (time+1)
f1 = f1 / (time+1)
loss = loss.item()
f1 = f1.item()
print("Loss_test: {:.4f}".format(loss))
print("F1_test: {:.4f}".format(f1))
