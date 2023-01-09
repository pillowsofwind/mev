import os
import requests

URL = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v2"
UNISWAPV2_FILE = os.path.join(os.path.dirname(__file__), "../data/uniswap_v2_addr_list.ctsv")


def get_query(query):
    r = requests.post(URL, json={'query': query})
    if r.status_code == 200:
        return r.json()
    else:
        raise Exception('Query failed and return code is {}.      {}'.format(r.status_code, query))


def main():

    raw_query  = """query MyQuery {
  pairs(first:LIMIT, orderBy: createdAtTimestamp, orderDirection: asc, where: {createdAtTimestamp_gt: TIMESTAMP}) {
    id
    token0Price
    token0 {
      id
      name
      symbol
      decimals
    }
    token1Price
    token1 {
      id
      name
      symbol
      decimals
    }
    createdAtBlockNumber
    createdAtTimestamp
  }
}
    """
    if not os.path.exists(UNISWAPV2_FILE):
        with open(UNISWAPV2_FILE, "w") as f:
            columns = ['address', 'token0_address', 'token0_name', 'token0_symbol', 'token0_decimals', 'token0_price', 'token1_address', 'token1_name', 'token1_symbol', 'token1_decimals', 'token1_price', 'createdAtBlockNumber', 'createdAtTimestamp']
            f.write(",\t".join(columns) + "\n")
        max_createdAtTimestamp = 0
    else:
        with open(UNISWAPV2_FILE, "r") as f:
            lines = f.readlines()
            line = lines[-1].strip()
            
        max_createdAtTimestamp = int(line.split(",")[-1])
        print("max_createdAtTimestamp: ", max_createdAtTimestamp)

    LIMIT = 1000

    while True:
        query = raw_query.replace("LIMIT", str(LIMIT)).replace("TIMESTAMP", str(max_createdAtTimestamp))
        result = get_query(query)

        while "data" not in result or "pairs" not in result["data"]:
            result = get_query(query)

        for pair in result["data"]["pairs"]:
            pair_address = pair["id"]

            token0 = pair["token0"]
            token0_address = token0["id"]
            token0_name = token0["name"]
            token0_symbol = token0["symbol"]
            token0_decimals = token0["decimals"]
            token0_price = pair["token0Price"]

            token1 = pair["token1"]
            token1_address = token1["id"]
            token1_name = token1["name"]
            token1_symbol = token1["symbol"]
            token1_decimals = token1["decimals"]
            token1_price = pair["token1Price"]

            createdAtBlockNumber = pair["createdAtBlockNumber"]
            createdAtTimestamp = pair["createdAtTimestamp"]

            max_createdAtTimestamp = max(max_createdAtTimestamp, int(createdAtTimestamp))

            with open(UNISWAPV2_FILE, "a") as f:
                f.write(",\t".join([pair_address, token0_address, token0_name, token0_symbol, str(token0_decimals), str(token0_price), token1_address, token1_name, token1_symbol, str(token1_decimals), str(token1_price), str(createdAtBlockNumber), str(createdAtTimestamp)]) + "\n")

        print("max_createdAtTimestamp: ", max_createdAtTimestamp)

        if len(result["data"]["pairs"]) < LIMIT:
            break



if __name__ == "__main__":
    main()
