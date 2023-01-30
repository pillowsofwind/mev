from config import *
from dataset import *
from dfs import *

import time

UNISWAPV2_PAIRS_FILE = "../data/all_uniswap_v2_data.csv"
uniswap_v2_pairs = load_uniswap_v2_pairs(UNISWAPV2_PAIRS_FILE, ["USD"])


def find_arb_in_a_block(block_number, tokenIn, tokenOut, maxHops, maxFound=10):
    path = [tokenIn]
    pairs = uniswap_v2_pairs[block_number]
    current_pairs = []
    best_trades = []

    start_time = time.time()

    trades = findArb(pairs, tokenIn, tokenOut, maxHops, current_pairs, path, best_trades, maxFound)

    end_time = time.time()

    print("Time taken: ", end_time - start_time)

    return trades


def main():
    trades = find_arb_in_a_block(16516000, WETH, WETH, 3)
    print(trades)


if __name__ == "__main__":
    main()
