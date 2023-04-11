from config import *
from dataset import *
from dfs import *

import time

uniswap_v2_pairs = load_uniswap_v2_pairs_v2("../arb_data/pools.csv")

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
    for i in range(16086234, 16093396+1):
        for t in STABLE_COINS:
            trades = find_arb_in_a_block(i, t, t, 3)
            if len(trades) > 0 and trades[0]["p"] > 5:
                print(i, t, trades)
                break
        if len(trades) > 0 and trades[0]["p"] > 5:
            break

if __name__ == "__main__":
    main()
