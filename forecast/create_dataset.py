import os
import sys

parentdir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir)

from arb_data import get_cpmm_arb_txn as carb
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

def test_cpmm_arb():
    carb.get_cpmm_data('../arb_data/one_day_arb.csv')
    carb.generate_graph_coo('../arb_data/one_day_arb.csv','../arb_data/pool_info.csv')
    carb.generate_graph_input('../arb_data/one_day_arb.csv','../arb_data/pools.csv',16086270)
    carb.generate_graph_output('../arb_data/one_day_arb.csv','../arb_data/pools.csv',16090709,30)

# COO format of graph, represent the edges
def get_edge_index():
    # graph_coo=carb.generate_graph_coo('../arb_data/one_day_arb.csv','../arb_data/pool_info.csv')
    # return torch.tensor(graph_coo,dtype=torch.long)
    return carb.generate_graph_coo('../arb_data/one_day_arb.csv','../arb_data/pool_info.csv')

# get node features x (input)
def get_x(blocknumber):
    # x=carb.generate_graph_input('../arb_data/one_day_arb.csv','../arb_data/pools.csv',blocknumber)
    # return torch.tensor(x,dtype=torch.float)
    return carb.generate_graph_input('../arb_data/one_day_arb.csv','../arb_data/pools.csv',blocknumber)

# get node targets y (output)
def get_y(blocknumber, period=50):
    # y=carb.generate_graph_output('../arb_data/one_day_arb.csv','../arb_data/pools.csv',blocknumber,period)
    # return torch.tensor(y,dtype=torch.long)
    return carb.generate_graph_output('../arb_data/one_day_arb.csv','../arb_data/pools.csv',blocknumber,period)



class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

        # load data from the save path using `process`
        self.data, self.slices = torch.load(self.processed_paths[0])

    # return raw data file names
    @property
    def raw_file_names(self):
        return ['./one_day_arb.csv']

    # return the name when saving the dataset files with method `process`
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    # no need, this is for downloading dataset from the web
    # def download(self):
    #     # Download to `self.raw_dir`.
    #     download_url(url, self.raw_dir)
    #     ...

    # methods to create the dataset
    def process(self):
        
        # Read data into huge PyG `Data` list.
        start_block_num = 16086234
        end_block_num = 16093396
        duration = 50

        data_list = []

        graph_edges=get_edge_index()

        for bn in tqdm(range(start_block_num,end_block_num,duration)):
            data = Data(edge_index=graph_edges,x=get_x(bn),y=get_y(bn,duration))
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



# test_cpmm_arb()
# blocknumber=16086270
# data = Data(edge_index=get_edge_index(),x=get_x(blocknumber),y=get_y(blocknumber))

# print(data)

# dataset=MyOwnDataset('dataset')
# process data and save
# print('process finished!')