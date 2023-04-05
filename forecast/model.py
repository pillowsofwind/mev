import create_dataset 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal import StaticGraphTemporalSignal
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.nn.recurrent import GConvGRU
from tqdm import tqdm


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, filters):
        super(RecurrentGCN, self).__init__()
        self.recurrent = GConvGRU(node_features, filters, 2)
        self.linear = torch.nn.Linear(filters, 1)

    def forward(self, x, edge_index):
        # x = num_nodes * 630 (original features dim)
        h = self.recurrent(x, edge_index)
        h = F.relu(h)
        h = self.linear(h)
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



model = RecurrentGCN(node_features=630, filters=32)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()

for epoch in tqdm(range(100)):
    cost = 0
    for time, snapshot in enumerate(train_dataset):
        y_hat = model(snapshot.x, snapshot.edge_index)
        cost = torch.mean((y_hat-snapshot.y)**2)
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()
        

torch.save(model, './model/GConvGRU.pkl') 

# evaluate the model
model.eval()
cost = 0
for time, snapshot in enumerate(test_dataset):
    y_hat = model(snapshot.x, snapshot.edge_index)
    cost = cost + torch.mean((y_hat-snapshot.y)**2)
cost = cost / (time+1)
cost = cost.item()
print("MSE: {:.4f}".format(cost))