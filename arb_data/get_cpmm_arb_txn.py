import pandas as pd
import numpy as np
import time
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
        if len(pools) == len(new_protocols):
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
    for item in records:
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


# get a gold-ordered list of tokens used in cpmm according to the arbs
def get_cpmm_token_list_from_arbs(arbs):
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


# get pool token adresses from a pool info dataframe object
def get_pool_tokens(df,pool):
    assert(df.loc[(df['poolAddress'] == pool)].empty!=True)
    addrs=df.loc[(df['poolAddress'] == pool)]['token addresses'].values.astype(str)
    return addrs


# generate the graph adjacency matrix
# dim=num_nodes*num_nodes
def create_graph_adjacency_matrix(arbs,pool_info):
    t_s=time.time()
    pool_addrs=get_cpmm_pool_list_from_arbs(arbs)
    num_nodes=len(pool_addrs)
    print('number of nodes:{:d}'.format(num_nodes))
    df_pool_info = pd.read_csv(pool_info)
    adj_mat=np.zeros([num_nodes,num_nodes],dtype=int)

    # fill the adjacency matrix
    # print(pool_addrs[0])
    # print(get_pool_tokens(df_pool_info,pool_addrs[0]))
    for i in range(num_nodes):
        print('\rprogress={:.3f}%'.format((i+1)*100/num_nodes),end="")
        for j in range(i, num_nodes):
            if i != j:
                addrs_i=get_pool_tokens(df_pool_info,pool_addrs[i])
                addrs_j=get_pool_tokens(df_pool_info,pool_addrs[j])
                tmp = np.intersect1d(addrs_i,addrs_j)
                if len(tmp)>0:
                    adj_mat[i][j]=1
                    adj_mat[j][i]=1

    t_e=time.time()
    print('time cost generating matrix=%.4fs'%(t_e-t_s))
    return adj_mat


# generate graph signals of the graph at a certain blocknumber
# dim=num_nodes, each element is an attributes vector
# an attribute vector looks like [onehot_token0,onehot_token1,volume_token0,volume_token1]
# onehot_tokenx is the onehot vector representation of a token
def generate_graph_signal(arbs,pools,blocknumber):
    pool_addrs=get_cpmm_pool_list_from_arbs(arbs)
    num_nodes=len(pool_addrs)
    df_pools=pd.read_csv(pools)


# generate graph output (labels) 
# mark arbitrage pools in a period [blocknumber, blocknumber+duration)
# dim=num_nodes, each element is either 0 or 1 (an arbitrage path went through it)
def generate_graph_output(arbs,pools,blocknumber,duration=50):


get_cpmm_data('./one_day_arb.csv')
# pools_to_pool_info('./pools.csv','pool_info.csv')
create_graph_adjacency_matrix('./one_day_arb.csv','pool_info.csv')
generate_graph_signal('./one_day_arb.csv','./pools.csv',16086270)
generate_graph_output('./one_day_arb.csv','./pools.csv',16086270,30)
