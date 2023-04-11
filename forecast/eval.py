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
from sklearn.metrics import f1_score
import random


num_nodes = 307
num_tokens = 157


class RecurrentGCN(torch.nn.Module):
    def __init__(self, filters):
        super(RecurrentGCN, self).__init__()
        torch.manual_seed(1234567)
        self.encode_linear=nn.Linear(num_tokens,4)
        self.recurrent = GConvGRU(10, filters, 2)
        self.linear = torch.nn.Linear(filters, 1)

    def forward(self, x, edge_index):
        # x = num_nodes * 630 (original features dim)
        v0 = x[:,0:num_tokens]
        v1 = x[:,num_tokens+1:2*num_tokens+1]
        v0 = self.encode_linear(v0)
        v1 = self.encode_linear(v1)
        a = torch.cat([v0,x[:,num_tokens:num_tokens+1],v1,x[:,2*num_tokens+1:2*num_tokens+2]],dim=1)
        h = self.recurrent(a, edge_index)
        h = F.relu(h)
        h = self.linear(h)
        h = torch.sigmoid(h)
        return h


model=torch.load('./model/GConvGRU.pkl')


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

train_dataset,test_dataset=temporal_signal_split(dataset,train_ratio=0.8)

criterion = nn.BCELoss()

# device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
# print('Evaluation device: ',device)

# # train_dataset = train_dataset.to(device)
# # test_dataset = test_dataset.to(device)

model = model.to("cpu")
# criterion = criterion.to(device)

train_dataset_snapshots = [s for s in train_dataset]
test_dataset_snapshots = [s for s in test_dataset]
# zero_tensor=torch.zeros(1).to(device)
# none_tensor=torch.tensor(0).to(device)


def round_tensor(t, decimal_places=3):
    return round(t.item(), decimal_places)


# evaluate the model
model.eval()
loss = 0
f1 = 0
f1_random = 0
pred_rand=[random.random()>=0.5 for i in range(num_nodes)]
for time, snapshot in enumerate(test_dataset_snapshots):
    y_hat = model(snapshot.x, snapshot.edge_index)
    loss += criterion(y_hat,snapshot.y.float())
    pred = y_hat.ge(.5).view(-1)
    print(sum(pred.numpy()))
    print(sum(snapshot.y.numpy()))
    f1 += f1_score(pred.numpy(),snapshot.y.numpy(),zero_division=0)
    f1_random += f1_score(pred_rand,snapshot.y.numpy(),zero_division=0)
loss = loss / (time+1)
loss = loss.item()
f1 = f1 / (time+1)
f1_random = f1_random / (time+1)
print("Loss_test: {:.4f}".format(loss))
print("F1_test: {:.4f}".format(f1))
print("F1_random: {:.4f}".format(f1_random))
