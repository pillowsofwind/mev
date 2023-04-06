import pandas as pd
import numpy as np
from tqdm import tqdm
import math

start_block_num = 16086234
end_block_num = 16093396


# quick peek of one day's cpmm arb data
def get_cpmm_data(arbs):
    df = pd.read_csv(arbs)
    records = df.to_dict('records')

    cpmm_txns = []
    cpmm_pool_addrs = []
    cpmm_tokens_addrs = []
    for item in records:
        # print(item['blockNumber'])
        # assert(len(item['token_symbols'].split(' '))==len(item['token_addresses'].split(' ')))
        pools = item['pools'].split(' ')
        protocols = []
        for p in pools:
            if '(' not in p:
                continue
            protocols.append(p.split('(')[1][:-1:])
        new_protocols = list(filter(lambda x: x in ['sush', 'usp2'], protocols))
        if len(pools) == len(new_protocols): # this suggest the pool is a cpmm pool
            cpmm_txns.append(item)
            tokens = item['token_addresses'].split(' ') 
            for tok in tokens:
                if tok not in cpmm_tokens_addrs:
                    cpmm_tokens_addrs.append(tok)
            pools = item['pool_addresses'].split(' ')
            for p in pools:
                if p not in cpmm_pool_addrs:
                    cpmm_pool_addrs.append(p)

    print('number of blocks in this day: %d'%(end_block_num-start_block_num+1))
    print('number of cpmm arb txns: %d'%(len(cpmm_txns)))
    print('number of cpmm tokens: %d'%(len(cpmm_tokens_addrs)))
    print('number of cpmm pools: %d'%(len(cpmm_pool_addrs)))


# generate pool info from pools
def pools_to_pool_info(pools,pool_info):
    df = pd.read_csv(pools)
    records = df.to_dict('records')
    outdf = pd.DataFrame(columns=['poolAddress','token addresses'])

    pools=[]
    for item in tqdm(records):
        pool_addr=item['poolAddress'].lower()
        if pool_addr not in pools:
            pools.append(pool_addr)
            outdf.loc[len(outdf.index)]=[pool_addr,'%s %s'%(item['token0Address'].lower(),item['token1Address'].lower())]
    outdf.to_csv(pool_info,index=False,sep=',')
    print('number of cpmm pools: %d'%(len(pools)))


# get a gold-ordered list (the order of the pools is consistent with the graph adjacency matrix, the graph signal, and the graph output)
# according to the arbs
def get_cpmm_pool_list_from_arbs(arbs):
    df = pd.read_csv(arbs)
    records = df.to_dict('records')

    cpmm_pool_addrs = []
    for item in records:
        pools = item['pools'].split(' ')
        protocols = []
        for p in pools:
            if '(' not in p:
                continue
            protocols.append(p.split('(')[1][:-1:])
        new_protocols = list(filter(lambda x: x in ['sush', 'usp2'], protocols))
        if len(pools) == len(new_protocols):
            pools = item['pool_addresses'].split(' ')
            for p in pools:
                if p not in cpmm_pool_addrs:
                    cpmm_pool_addrs.append(p)
    return cpmm_pool_addrs


# get a gold-ordered list of tokens used in cpmm according to the arbs
def get_cpmm_token_list_from_arbs(arbs):
    df = pd.read_csv(arbs)
    records = df.to_dict('records')

    cpmm_tokens_addrs = []
    for item in records:
        pools = item['pools'].split(' ')
        protocols = []
        for p in pools:
            if '(' not in p:
                continue
            protocols.append(p.split('(')[1][:-1:])
        new_protocols = list(filter(lambda x: x in ['sush', 'usp2'], protocols))
        if len(pools) == len(new_protocols):
            tokens = item['token_addresses'].split(' ') 
            for tok in tokens:
                if tok not in cpmm_tokens_addrs:
                    cpmm_tokens_addrs.append(tok)
    return cpmm_tokens_addrs
    

# get the one-hot representation of a token
def get_token_onehot(gold_token_list,token):
    dim=len(gold_token_list)
    assert(gold_token_list.count(token)==1)
    index=gold_token_list.index(token) # this would return ValueError is token not found
    onehot=np.zeros(dim)
    onehot[index]=1
    return onehot


# get pool token adresses from a pool info dataframe object
def get_pool_tokens(df,pool):
    assert(df.loc[(df['poolAddress'] == pool)].empty!=True)
    addrs=df.loc[(df['poolAddress'] == pool)]['token addresses'].values.astype(str)
    addrs=addrs[0].split(' ')
    return addrs


'''Methods below are used to get graph related data, return values must be ndarray'''


# generate the graph adjacency matrix (sparse)
# dim=num_pools*num_pools
def create_graph_adjacency_matrix(arbs,pool_info):
    print('generating graph adjacency matrix...')
    pool_addrs=get_cpmm_pool_list_from_arbs(arbs)
    num_pools=len(pool_addrs)
    print('number of nodes:{:d}'.format(num_pools))
    df_pool_info = pd.read_csv(pool_info)
    graph_adj_mat=np.zeros([num_pools,num_pools],dtype=int)

    # fill the adjacency matrix
    # print(pool_addrs[0])
    # print(get_pool_tokens(df_pool_info,pool_addrs[0]))
    for i in tqdm(range(num_pools)):
        # print('\rprogress={:.3f}%'.format((i+1)*100/num_pools),end="")
        for j in range(i, num_pools):
            if i != j:
                addrs_i=get_pool_tokens(df_pool_info,pool_addrs[i])
                addrs_j=get_pool_tokens(df_pool_info,pool_addrs[j])
                tmp = np.intersect1d(addrs_i,addrs_j)
                if len(tmp)>0:
                    graph_adj_mat[i][j]=1
                    graph_adj_mat[j][i]=1

    return graph_adj_mat


# generate the graph in COO format (dense)
# dim=2*num_edges (there is an edge if two pools have the same token)
def generate_graph_coo(arbs,pool_info):
    print('generating graph COO...')
    pool_addrs=get_cpmm_pool_list_from_arbs(arbs)
    num_pools=len(pool_addrs)
    print('number of nodes:{:d}'.format(num_pools))
    df_pool_info = pd.read_csv(pool_info)

    edge_start=[]
    edge_end=[]

    for i in tqdm(range(num_pools)):
        # print('\rprogress={:.3f}%'.format((i+1)*100/num_pools),end="")
        for j in range(i, num_pools):
            if i != j:
                addrs_i=get_pool_tokens(df_pool_info,pool_addrs[i])
                addrs_j=get_pool_tokens(df_pool_info,pool_addrs[j])
                tmp = np.intersect1d(addrs_i,addrs_j)
                if len(tmp)>0: # there is an edge(i,j)
                    edge_start.append(i)
                    edge_end.append(j)
                    edge_start.append(j)
                    edge_end.append(i)
    
    graph_coo=(edge_start,)+(edge_end,)
    graph_coo=np.array(graph_coo)
    print('number of edges in graph: {:d}'.format(len(graph_coo[1])))
    return graph_coo


# generate graph input (signals) of the graph at a certain blocknumber
# dim=num_pools, each element is an feature vector
# an feature vector looks like [onehot_token0,volume_token0,onehot_token1,volume_token1] with dim=2*(onehot_dim+1)
# onehot_tokenx is the onehot vector representation of a token
def generate_graph_input(arbs,pools,blocknumber):
    # print('generating graph input...')
    pool_addrs=get_cpmm_pool_list_from_arbs(arbs)
    token_addrs=get_cpmm_token_list_from_arbs(arbs)
    num_pools=len(pool_addrs)
    num_tokens=len(token_addrs) # dimension of the one-hot representation of a token
    df_pools=pd.read_csv(pools)

    largest_volume=0
    graph_input=np.zeros([num_pools,2*num_tokens+2],dtype=float)
    pool_state_info=df_pools.loc[(df_pools['blockNumber']==blocknumber)].to_dict('records')
    for pool_info in pool_state_info:
        token0=get_token_onehot(token_addrs,pool_info['token0Address'])
        token1=get_token_onehot(token_addrs,pool_info['token1Address'])
        token0_volume=[float(pool_info['balance0'])]
        token1_volume=[float(pool_info['balance1'])]
        largest_volume=max(largest_volume,token0_volume[0],token1_volume[0])
        graph_input[pool_addrs.index(pool_info['poolAddress'])]=np.concatenate((token0,token0_volume,token1,token1_volume))

    for node_input in graph_input:
        node_input[num_tokens]=node_input[num_tokens]/largest_volume
        node_input[2*num_tokens+1]=node_input[2*num_tokens+1]/largest_volume
    # print('shape of the graph input signal: ',graph_input.shape)
    # print(graph_input)
    return graph_input


# generate graph output (labels) 
# mark arbitrage pools in a period [blocknumber, blocknumber+duration)
# dim=num_pools, each element is either 0 or 1 (an arbitrage path went through it)
def generate_graph_output(arbs,pools,blocknumber,duration=50):
    # print('generating graph output...')
    pool_addrs=get_cpmm_pool_list_from_arbs(arbs)
    df_arbs=pd.read_csv(arbs)

    arbs_start=df_arbs.loc[(df_arbs['blockNumber']>=blocknumber)].to_dict('records')

    graph_output=np.zeros([len(pool_addrs),1],dtype=int)
    for arb in arbs_start:
        if arb['blockNumber']>=(blocknumber+duration):
            break
        pools=arb['pools'].split(' ')
        protocols=[]
        for p in pools:
            if '(' not in p:
                continue
            protocols.append(p.split('(')[1][:-1:])
        new_protocols=list(filter(lambda x: x in ['sush', 'usp2'], protocols))
        if len(pools) == len(new_protocols):
            p_addrs = arb['pool_addresses'].split(' ')
            for p_addr in p_addrs: # all the pools that involve in thi arb
                idx=pool_addrs.index(p_addr)
                graph_output[idx]=1

    # print(graph_output)
    return graph_output


# df=pd.read_csv('./pools.csv')

# df["poolAddress"] = df["poolAddress"].apply(lambda x: x.lower())
# df["token0Address"] = df["token0Address"].apply(lambda x: x.lower())
# df["token1Address"] = df["token1Address"].apply(lambda x: x.lower())

# df.to_csv('./pools.csv',index=False)

# pools_to_pool_info('./pools.csv','pool_info.csv')

# get_cpmm_data('./one_day_arb.csv')
# create_graph_adjacency_matrix('./one_day_arb.csv','pool_info.csv')
# generate_graph_coo('./one_day_arb.csv','pool_info.csv')
# generate_graph_input('./one_day_arb.csv','./pools.csv',16086270)
# generate_graph_output('./one_day_arb.csv','./pools.csv',16090709,30)
