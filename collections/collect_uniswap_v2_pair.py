import json
import os
import time
import pandas as pd
import requests

from web3 import Web3, HTTPProvider

w3 = Web3(HTTPProvider("http://localhost:8545"))

UNISWAPV2_FILE = os.path.join(os.path.dirname(__file__), "../data/uniswap_v2_addr_list.ctsv")
UNISWAPV2_PAIRS_FILE = os.path.join(os.path.dirname(__file__), "../data/uniswap_v2_pairs_list_new.csv")

URL = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v2"

STABLE_COINS = ["DAI", "USDC", "USDT", "BUSD"]

RAW_QUERY  = """query MyQuery {
    pairs(first:LIMIT, where: {id_in: ADDR_LIST}, block: {number: BLOCK_NUMBER}) {
        reserve0
        reserve1
        id
    }
}
"""

def get_query(query):
    r = requests.post(URL, json={'query': query})
    if r.status_code == 200:
        return r.json()
    else:
        raise Exception('Query failed and return code is {}.      {}'.format(r.status_code, query))


df = pd.read_csv(UNISWAPV2_FILE, sep=",\t")

stable_coin_pairs = df[(df["token0_symbol"].isin(STABLE_COINS)) | (df["token1_symbol"].isin(STABLE_COINS))]

sample_of_stable_coin_pairs = stable_coin_pairs.head(6000)

symbols = set()

for token0_symbol, token1_symbol in sample_of_stable_coin_pairs[["token0_symbol", "token1_symbol"]].values:
    symbols.add(token0_symbol)
    symbols.add(token1_symbol)

symbols = list(symbols)

supplement_stable_coin_pairs = df[(df["token0_symbol"].isin(symbols)) & (df["token1_symbol"].isin(symbols))]
pairs_addresses = supplement_stable_coin_pairs["address"].tolist()


def query_from_uniswap_v2(addr_list, block_number, limit=1000):
    query = RAW_QUERY.replace("ADDR_LIST", json.dumps(addr_list)).replace("BLOCK_NUMBER", str(block_number)).replace("LIMIT", str(limit))
    result = get_query(query)

    while "data" not in result or "pairs" not in result["data"]:
        result = get_query(query)

    return result

if not os.path.exists(UNISWAPV2_PAIRS_FILE):
    with open(UNISWAPV2_PAIRS_FILE, "w") as f:
        f.write("blockNumber,pairAddress,reserve0,reserve1\n")
    block_start_number = 163926000
else:
    df = pd.read_csv(UNISWAPV2_PAIRS_FILE)
    block_start_number = df["blockNumber"].max() + 1

count = 0

while True:
    current_block_number = int(w3.eth.get_block_number())

    if current_block_number == block_start_number:
        time.sleep(12)
        continue

    for block_number in range(block_start_number, current_block_number):
        for i in range(0, len(pairs_addresses), 1000):
            upper_bound = min(i + 1000, len(pairs_addresses))
            addr_list = pairs_addresses[i:upper_bound] 
            result = query_from_uniswap_v2(addr_list, block_number)

            for pair in result["data"]["pairs"]:
                pair_address = pair["id"]
                reserve0 = pair["reserve0"]
                reserve1 = pair["reserve1"]

                with open(UNISWAPV2_PAIRS_FILE, "a") as f:
                    f.write("{},{},{},{}\n".format(block_number, pair_address, reserve0, reserve1))

        print("block_number: {}, i: {}, upper_bound: {}".format(block_number, i, upper_bound))

    block_start_number = current_block_number

    count += 1
    if count >= 10000:
        break