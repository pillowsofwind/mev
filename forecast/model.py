import create_dataset 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal import StaticGraphTemporalSignal
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.nn.recurrent import GConvGRU
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt



class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, filters):
        super(RecurrentGCN, self).__init__()
        self.recurrent = GConvGRU(10, filters, 2)
        self.linear = torch.nn.Linear(filters, 1)

    def forward(self, x, edge_index):
        # x = num_nodes * 630 (original features dim)
        m = nn.Linear(314,4)
        v0 = x[:,0:314]
        v1 = x[:,315:629]
        v0 = m(v0)
        v1 = m(v1)
        a = torch.cat([v0,x[:,314:315],v1,x[:,629:630]],dim=1)
        h = self.recurrent(a, edge_index)
        h = F.relu(h)
        h = self.linear(h)
        h = torch.sigmoid(h)
        return h
    

# this is a list of PyG `Data` objects
data_list=create_dataset.MyOwnDataset('dataset')

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


x_data=[]
y_data=[]
def round_tensor(t, decimal_places=3):
    return round(t.item(), decimal_places)

model = RecurrentGCN(node_features=630, filters=32)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()

for epoch in tqdm(range(100)):
    cost = 0
    for time, snapshot in enumerate(train_dataset):
        y_hat = model(snapshot.x, snapshot.edge_index)
        loss = nn.BCELoss()
        cost = cost + loss(y_hat,snapshot.y.float())
    cost = cost / (time+1)
    y_data.append(round_tensor(cost))
    x_data.append(epoch)
    cost.backward()
    optimizer.step()
    optimizer.zero_grad()

plt.plot(x_data,y_data,color='r')
plt.xlabel('epoches')
plt.ylabel('BCE loss')
# plt.show()
plt.savefig('BCE.jpg')

torch.save(model, './model/GConvGRU.pkl') 

# evaluate the model
model.eval()
cost = 0
for time, snapshot in enumerate(test_dataset):
    y_hat = model(snapshot.x, snapshot.edge_index)
    loss = nn.BCELoss()
    cost = cost + loss(y_hat,snapshot.y.float())
cost = cost / (time+1)
cost = cost.item()
print("BCE: {:.4f}".format(cost))


# do nothing
zero_tensor = torch.from_numpy(np.zeros([489,1]))

orignal = 0
for time, snapshot in enumerate(test_dataset):
    loss = nn.BCELoss()
    orignal = orignal + loss(zero_tensor,snapshot.y.double())
orignal = orignal / (time+1)
orignal = orignal.item()
print("Original BCE: {:.4f}".format(orignal))