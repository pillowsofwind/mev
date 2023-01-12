import os
import pandas as pd
from collections import defaultdict


UNISWAPV2_FILE = os.path.join(os.path.dirname(__file__), "../data/uniswap_v2_addr_list.ctsv")
UNISWAPV2_PAIRS_FILE = os.path.join(os.path.dirname(__file__), "../data/uniswap_v2_pairs_list.csv")


uniswap_v2_df = pd.read_csv(UNISWAPV2_FILE, sep=",\t")
uniswap_v2_pairs_df = pd.read_csv(UNISWAPV2_PAIRS_FILE)

pair_details = defaultdict(dict)
for address, token0_address, token0_symbol, token0_decimals, token1_address, token1_symbol, token1_decimals in uniswap_v2_df[["address", "token0_address", "token0_symbol", "token0_decimals", "token1_address", "token1_symbol", "token1_decimals"]].values:
    try:
        pair_details[address]["token0"] = {
            "address": token0_address,
            "symbol": token0_symbol,
            "decimals": int(token0_decimals)
        }

        pair_details[address]["token1"] = {
            "address": token1_address,
            "symbol": token1_symbol,
            "decimals": int(token1_decimals)
        }
    except:
        continue


uniswap_v2_pairs = defaultdict(list)

for block_number, pair_address, reserve0, reserve1 in uniswap_v2_pairs_df[["blockNumber", "pairAddress", "reserve0", "reserve1"]].values:
    pair = {
        "address": pair_address,
        "token0": pair_details[pair_address]["token0"],
        "token1": pair_details[pair_address]["token1"],
        "reserve0": int(reserve0 * 1e18),
        "reserve1": int(reserve1 * 1e18)
    }
    uniswap_v2_pairs[block_number].append(pair)
