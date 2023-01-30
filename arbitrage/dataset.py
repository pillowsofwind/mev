import os
import pandas as pd
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
UNISWAPV2_FILE = os.path.join(DATA_DIR, "uniswap_v2_addr_list.csv")
UNISWAP_V2_DF = pd.read_csv(UNISWAPV2_FILE)
SUSHISWAP_FILE = os.path.join(DATA_DIR, "sushiswap_addr_list.csv")
SUSHISWAP_DF = pd.read_csv(SUSHISWAP_FILE)
SHIBASWAP_FILE = os.path.join(DATA_DIR, "shibaswap_addr_list.csv")
SHIBASWAP_DF = pd.read_csv(SHIBASWAP_FILE)
PAIR_DETAILS = defaultdict(dict)

for df in [UNISWAP_V2_DF, SUSHISWAP_DF, SHIBASWAP_DF]:
    for address, token0_address, token0_symbol, token0_decimals, token1_address, token1_symbol, token1_decimals in df[["address", "token0_address", "token0_symbol", "token0_decimals", "token1_address", "token1_symbol", "token1_decimals"]].values:
        # if type(address) != str or token0_decimals is None or token1_decimals is None:
        #     continue
        address = address.lower()
        try:
            PAIR_DETAILS[address]["token0"] = {
                "address": token0_address,
                "symbol": token0_symbol,
                "decimals": int(token0_decimals)
            }

            PAIR_DETAILS[address]["token1"] = {
                "address": token1_address,
                "symbol": token1_symbol,
                "decimals": int(token1_decimals)
            }
        except:
            if address in PAIR_DETAILS:
                del PAIR_DETAILS[address]
            continue


def load_uniswap_v2_pairs(path, filter_tokens=[]):
    uniswap_v2_pairs_df = pd.read_csv(path)
    uniswap_v2_pairs = defaultdict(list)

    for block_number, pair_address, reserve0, reserve1 in uniswap_v2_pairs_df[["block_number", "address", "reserve0", "reserve1"]].values:
        pair_address = pair_address.lower()

        if pair_address not in PAIR_DETAILS:
            continue

        token0 = PAIR_DETAILS[pair_address]["token0"]
        token1 = PAIR_DETAILS[pair_address]["token1"]

        token_symbols = set([token0["symbol"], token1["symbol"]])
        filter_token_symbols = set(filter_tokens)

        if len(filter_token_symbols.intersection(token_symbols)) > 0:
            continue

        pair = {
            "address": pair_address,
            "token0": token0,
            "token1": token1,
            "reserve0": int(reserve0),
            "reserve1": int(reserve1)
        }
        uniswap_v2_pairs[block_number].append(pair)

    return uniswap_v2_pairs
