import os
import json
import pandas as pd

from web3 import Web3, HTTPProvider

ABI_PATH = os.path.join(os.path.dirname(__file__), "../abi")
UNISWAP_V2_FACTORY_ABI_PATH = os.path.join(ABI_PATH, "UniswapV2Factory.json")
UNISWAP_V2_PAIR_ABI_PATH = os.path.join(ABI_PATH, "IUniswapV2Pair.json")
ERC20_ABI_PATH = os.path.join(ABI_PATH, "IERC20.json")
ERC20_OLD_ABI_PATH = os.path.join(ABI_PATH, "IERC20.old.json")

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data")
SUSHISWAP_PAIRS_PATH = os.path.join(DATA_PATH, "sushiswap_addr_list.csv")
SHIBASWAP_PAIRS_PATH = os.path.join(DATA_PATH, "shibaswap_addr_list.csv")


SUSHISWAP_FACTORY_ADDRESS = "0xC0AEe478e3658e2610c5F7A4A2E1777cE9e4f2Ac"
SHIBASWAP_FACTORY_ADDRESS = "0x115934131916C8b277DD010Ee02de363c09d037c"

w3 = Web3(HTTPProvider('http://127.0.0.1:8545'))

with open(UNISWAP_V2_FACTORY_ABI_PATH) as f:
    UNISWAP_V2_FACTORY_ABI = json.load(f)

with open(UNISWAP_V2_PAIR_ABI_PATH) as f:
    UNISWAP_V2_PAIR_ABI = json.load(f)

with open(ERC20_ABI_PATH) as f:
    ERC20_ABI = json.load(f)

with open(ERC20_OLD_ABI_PATH) as f:
    ERC20_OLD_ABI = json.load(f)


def dump_factory(factory_address):
    factory_contract = w3.eth.contract(address=factory_address, abi=UNISWAP_V2_FACTORY_ABI)
    all_pairs_length = factory_contract.functions.allPairsLength().call()
    pairs = []
    for i in range(all_pairs_length):
        pair_addr = factory_contract.functions.allPairs(i).call()
        pair_contract = w3.eth.contract(address=pair_addr, abi=UNISWAP_V2_PAIR_ABI)
        token0_addr = pair_contract.functions.token0().call()
        token1_addr = pair_contract.functions.token1().call()
        if w3.eth.get_code(token0_addr).hex() == "0x" or w3.eth.get_code(token1_addr).hex() == "0x":
            continue

        reserves = pair_contract.functions.getReserves().call()

        item = [i, pair_addr]
        print(i, pair_addr, token0_addr, token1_addr)
        for token_index, token_addr in enumerate([token0_addr, token1_addr]):
            try:
                token_contract = w3.eth.contract(address=token_addr, abi=ERC20_ABI)
                token_name = token_contract.functions.name().call()
                token_symbol = token_contract.functions.symbol().call()
                token_decimals = token_contract.functions.decimals().call()
            except:
                try:
                    token_contract = w3.eth.contract(address=token_addr, abi=ERC20_OLD_ABI)
                    token_name = token_contract.functions.name().call()
                    token_symbol = token_contract.functions.symbol().call()
                    token_name = token_name.decode('utf-8').strip("\x00")
                    token_symbol = token_symbol.decode('utf-8').strip("\x00")
                    token_decimals = token_contract.functions.decimals().call()
                except:
                    print("error", token_addr)
                    continue

            item += [token_addr, token_name, token_symbol, token_decimals, reserves[token_index]]

        pairs.append(tuple(item))

    pair_df = pd.DataFrame(pairs, columns=["index", "address", "token0_address", "token0_name", "token0_symbol", "token0_decimals", "reserve0", "token1_address", "token1_name", "token1_symbol", "token1_decimals", "reserve1"])

    return pair_df


sushiswap_df = dump_factory(SUSHISWAP_FACTORY_ADDRESS)
sushiswap_df.to_csv(SUSHISWAP_PAIRS_PATH, index=False)

shibaswap_df = dump_factory(SHIBASWAP_FACTORY_ADDRESS)
shibaswap_df.to_csv(SHIBASWAP_PAIRS_PATH, index=False)