import json
import os
import requests
import pandas as pd

URL = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v2"

LP_FILE = os.path.join(os.path.dirname(__file__), "../data/lp_addr_list.csv")

def get_query(query):
    r = requests.post(URL, json={'query': query})
    if r.status_code == 200:
        return r.json()
    else:
        raise Exception('Query failed and return code is {}.      {}'.format(r.status_code, query))


def main():
    df = pd.read_csv(LP_FILE)
    uniswap_v2_protocols = df[df["protocolName"] == "Uniswap V2"]
    uniswap_v2_protocols = uniswap_v2_protocols.head(100)

    addr_list = uniswap_v2_protocols["lpAddress"].tolist()

    raw_query  = """query MyQuery {
    pairs(where: {id_in: ADDR_LIST}, block: {number: BLOCK_NUMBER}) {
        reserve0
        reserve1
        id
    }
    }
    """

    # with open("output.csv", "w") as f:
        # f.write("blockNumber,pairAddress,reserve0,reserve1\n")

    for i in range(16336118, 16341001):
        print("block number: {}".format(i))
        
        query = raw_query.replace("ADDR_LIST", json.dumps(addr_list)).replace("BLOCK_NUMBER", str(i))

        result = get_query(query)
        while "data" not in result or "pairs" not in result["data"]:
            result = get_query(query)

        for pair in result["data"]["pairs"]:
            pair_address = pair["id"]
            reserve0 = pair["reserve0"]
            reserve1 = pair["reserve1"]

            with open("output.csv", "a") as f:
                f.write("{},{},{},{}\n".format(i, pair_address, reserve0, reserve1))

if __name__ == "__main__":
    main()
