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


num_nodes = 489


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, filters):
        super(RecurrentGCN, self).__init__()
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

loss = nn.BCELoss()

model = RecurrentGCN(node_features=630, filters=32)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Training device: ',device)

# train_dataset = train_dataset.to(device)
# test_dataset = test_dataset.to(device)

model = model.to(device)
loss = loss.to(device)

train_dataset_snapshots = [s.to(device) for s in train_dataset]
test_dataset_snapshots = [s.to(device) for s in test_dataset]


x_loss=[]
y_loss=[]
x_acc=[]
y_acc=[]


def cal_accuracy(y_pred, y_true):
    # to binary value {0,1}
    predicted = y_pred.ge(.5).view(-1)
    return (y_true == predicted).sum().float() / len(y_true)


def round_tensor(t, decimal_places=3):
    return round(t.item(), decimal_places)


model.train()

for epoch in tqdm(range(100)):
    cost = 0
    acc = 0
    for time, snapshot in enumerate(train_dataset_snapshots):
        y_hat = model(snapshot.x, snapshot.edge_index)
        acc += cal_accuracy(y_hat,snapshot.y)/num_nodes
        cost += loss(y_hat,snapshot.y.float())
    cost = cost / (time+1)
    acc = acc / (time+1)
    y_loss.append(round_tensor(cost))
    x_loss.append(epoch)
    y_acc.append(round_tensor(acc))
    x_acc.append(epoch)
    cost.backward()
    optimizer.step()
    optimizer.zero_grad()

plt.subplot(1,2,1)
plt.plot(x_loss,y_loss,color='r')
plt.xlabel('epoches')
plt.title('BCE LOSS')
plt.subplot(1,2,2)
plt.plot(x_acc,y_acc,color='g')
plt.xlabel('epoches')
plt.title('ACC')
# plt.show()
plt.savefig('train.PNG')

torch.save(model, './model/GConvGRU.pkl') 

# evaluate the model
model.eval()
cost = 0
acc = 0
for time, snapshot in enumerate(test_dataset_snapshots):
    y_hat = model(snapshot.x, snapshot.edge_index)
    cost = cost + loss(y_hat,snapshot.y.float())
    acc = acc + cal_accuracy(y_hat,snapshot.y)/num_nodes
cost = cost / (time+1)
acc = acc / (time+1)
cost = cost.item()
acc = acc.item()
print("BCE_test: {:.4f}".format(cost))
print("accuracy_test: {:.4f}".format(acc))


# predict nothing
zero_tensor = torch.from_numpy(np.zeros([num_nodes,1]))

orignal_BCE = 0
orignal_acc = 0
for time, snapshot in enumerate(test_dataset):
    orignal_BCE = orignal_BCE + loss(zero_tensor,snapshot.y.double())
    orignal_acc = orignal_acc + cal_accuracy(zero_tensor,snapshot.y.double())/num_nodes
orignal_BCE = orignal_BCE / (time+1)
orignal_acc = orignal_acc / (time+1)
orignal_BCE = orignal_BCE.item()
orignal_acc = orignal_acc.item()
print("Do nothing BCE: {:.4f}".format(orignal_BCE))
print("Do nothing accuracy: {:.4f}".format(orignal_acc))