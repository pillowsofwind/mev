import argparse
import json
import os
import pandas as pd

from web3 import Web3, HTTPProvider

ABI_PATH = os.path.join(os.path.dirname(__file__), "../abi")
UNISWAP_V2_PAIR_ABI_PATH = os.path.join(ABI_PATH, "IUniswapV2Pair.json")
LP_FILE = os.path.join(os.path.dirname(__file__), "../data/lp_addr_list.csv")

w3 = Web3(HTTPProvider('https://mainnet.infura.io/v3/df06acebf3db497aa87c4d9ffb0ee6d9'))

with open(UNISWAP_V2_PAIR_ABI_PATH) as f:
    UNISWAP_V2_PAIR_ABI = json.load(f)


def get_pair_reserves(pair_address, blockNumber='latest'):
    pair = w3.eth.contract(address=Web3.toChecksumAddress(pair_address), abi=UNISWAP_V2_PAIR_ABI)
    return pair.functions.getReserves().call(block_identifier=blockNumber)
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--startBlockNumber", type=int)
    parser.add_argument("--endBlockNumber", type=int)
    parser.add_argument("--output", type=str, default="output.csv")

    args = parser.parse_args()

    df = pd.read_csv(LP_FILE)
    uniswap_v2_protocols = df[df["protocolName"] == "Uniswap V2"]
    uniswap_v2_protocols = df.head(10)
    pair_states = []

    for i in range(args.startBlockNumber, args.endBlockNumber + 1):
        print("block number: {}".format(i))

        for pair_address, token0_address, token0_symbol, token1_address, token1_symbol in uniswap_v2_protocols[["lpAddress", "token0_address","token0_symbol","token1_address","token1_symbol"]].values:
            print("pair address: {}".format(pair_address))
            try:
                reserves = get_pair_reserves(pair_address, i)
            except:
                continue
            balance0 = reserves[0]
            balance1 = reserves[1]

            pair_state = {
                "blockNumber": i,
                "pairAddress": pair_address,
                "token0Address": token0_address,
                "token0Symbol": token0_symbol,
                "balance0": balance0,
                "token1Address": token1_address,
                "token1Symbol": token1_symbol,
                "balance1": balance1,
            }

            pair_states.append(pair_state)

    state_df = pd.DataFrame(pair_states)
    state_df.to_csv(args.output, index=False)