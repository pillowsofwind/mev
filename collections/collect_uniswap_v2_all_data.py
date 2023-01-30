import IPython
import json
import os
import pandas as pd
import threading
import random

import numpy as np
from multiprocessing.pool import ThreadPool
from web3 import Web3, HTTPProvider

random.seed(1024)
np.random.seed(1024)

pool = ThreadPool(50)

class MyThread(threading.Thread):
    def __init__(self, func, args):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception as e:
            print('thread exception:', e)
            return None

w3 = Web3(HTTPProvider("http://localhost:8545"))
 

ABI_PATH = os.path.join(os.path.dirname(__file__), "../abi")
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data")
UNISWAPV2_PAIRS_FILE = os.path.join(DATA_PATH, "uniswap_v2_addr_list.csv")
SUSHISWAP_PAIRS_FILE = os.path.join(DATA_PATH, "sushiswap_addr_list.csv")
SHIBASWAP_PAIRS_FILE = os.path.join(DATA_PATH, "shibaswap_addr_list.csv")
UNISWAP_V2_PAIR_ABI_PATH = os.path.join(ABI_PATH, "IUniswapV2Pair.json")

with open(UNISWAP_V2_PAIR_ABI_PATH) as f:
    UNISWAP_V2_PAIR_ABI = json.load(f)


STABLE_COINS = ["DAI", "USDC", "USDT", "BUSD"]
TARGETS = STABLE_COINS + ["WETH"]

uniswap_v2_df = pd.read_csv(UNISWAPV2_PAIRS_FILE)
sushiswap_df = pd.read_csv(SUSHISWAP_PAIRS_FILE)
shibaswap_df = pd.read_csv(SHIBASWAP_PAIRS_FILE)

symbols = set()

for index, df in enumerate([uniswap_v2_df, sushiswap_df, shibaswap_df]):
    if index == 0:
        uniswap_v2_df = uniswap_v2_df.head(140000)
    # print(index)
    for addr, reserve0, token0_symbol, token0_decimals, reserve1, token1_symbol, token1_decimals in df[["address", "reserve0", "token0_symbol", "token0_decimals", "reserve1", "token1_symbol", "token1_decimals"]].values:
        try:
            int(reserve1)
        except:
            continue
        token0_decimals = int(token0_decimals)
        token1_decimals = int(token1_decimals)

        if index == 0:
            reserve0 = float(reserve0)
            reserve1 = float(reserve1)
        else:
            reserve0 = int(reserve0)
            reserve0 /= pow(10, token0_decimals)
            reserve1 = int(reserve1)
            reserve1 /= pow(10, token1_decimals)
            
        if reserve0 < 1 or reserve1 < 1:
            continue
        
        if token0_symbol in TARGETS or token1_symbol in TARGETS:
            symbols.add(token0_symbol)
            symbols.add(token1_symbol)

for symbol in TARGETS:
    symbols.remove(symbol)

symbols = list(symbols)
random.shuffle(symbols)

selected_symbols = symbols

for symbol in TARGETS:
    selected_symbols.append(symbol)

pairs_addresses = []

for index, df in enumerate([uniswap_v2_df, sushiswap_df, shibaswap_df]):
    supplement_stable_coin_pairs = df[(df["token0_symbol"].isin(selected_symbols)) & (df["token1_symbol"].isin(selected_symbols))]
    for pair_addr, reserve0, token0_decimals, reserve1, token1_decimals in supplement_stable_coin_pairs[["address", "reserve0", "token0_decimals", "reserve1", "token1_decimals"]].values:
        try:
            int(reserve1)
        except:
            continue
        token0_decimals = int(token0_decimals)
        token1_decimals = int(token1_decimals)

        if index == 0:
            reserve0 = float(reserve0)
            reserve1 = float(reserve1)
        else:
            reserve0 = int(reserve0)
            reserve0 /= pow(10, token0_decimals)
            reserve1 = int(reserve1)
            reserve1 /= pow(10, token1_decimals)
        
        if reserve0 < 1 or reserve1 < 1:
            continue

        pairs_addresses.append(pair_addr)
    
# IPython.embed(colors="neutral")
        

def get_pair_reserves(pair_address, blockNumber='latest'):
    pair = w3.eth.contract(address=Web3.toChecksumAddress(pair_address), abi=UNISWAP_V2_PAIR_ABI)
    return pair.functions.getReserves().call(block_identifier=blockNumber)

def get_pairs_reserves(pairs_addresses, blockNumber='latest'):
    reserves = []
    count = 0
    for pair_address in pairs_addresses:
        count += 1
        if count % 20 == 0:
            print(count)
        try:
            pair_reserves = get_pair_reserves(pair_address, blockNumber)
            reserves.append((pair_address, pair_reserves[0], pair_reserves[1]))
        except Exception as e:
            print("error", pair_address, e)
            pass
    return reserves

pairs_addresses = [i for i in pairs_addresses if type(i) == str]

print(len(pairs_addresses))
block_number = 16516000

with open("../all_uniswap_v2_data.csv", "w") as f:
    f.write("block_number,address,reserve0,reserve1\n")

for i in range(block_number, block_number + 100):
    pair_limits = 250
    threads = []
    for j in range(0, len(pairs_addresses), pair_limits):
        pairs_addresses_sub = pairs_addresses[j:j+pair_limits]
        t = MyThread(func=get_pairs_reserves, args=(pairs_addresses_sub, i))
        t.start()
        threads.append(t)

    print(f"{len(threads)} thread start")
    for t in threads:
        t.join()
        ret = t.get_result()
    
        with open("./test2.csv", "a") as f:
            for pair_addr, pair_reserve0, pair_reserve1 in ret:
                f.write(f"{i},{pair_addr},{pair_reserve0},{pair_reserve1}\n")
        
    print(f"{i}")
