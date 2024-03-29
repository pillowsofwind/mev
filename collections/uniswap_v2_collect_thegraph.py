import os
import requests

URL = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v2"
UNISWAPV2_FILE = os.path.join(os.path.dirname(__file__), "../data/uniswap_v2_addr_list.csv")


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
    reserve0
    token0Price
    token0 {
      id
      name
      symbol
      decimals
    }
    reserve1
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
            columns = ['address', 'token0_address', 'token0_name', 'token0_symbol', 'token0_decimals', 'reserve0', 'token0_price', 'token1_address', 'token1_name', 'token1_symbol', 'token1_decimals', 'reserve1', 'token1_price', 'createdAtBlockNumber', 'createdAtTimestamp']
            f.write(",".join(columns) + "\n")
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
            token0_name = token0["name"].strip().replace(",", "(comma)")
            token0_symbol = token0["symbol"].strip().replace(",", "(comma)")
            token0_decimals = token0["decimals"]
            token0_price = pair["token0Price"]
            reserve0 = pair["reserve0"]

            token1 = pair["token1"]
            token1_address = token1["id"]
            token1_name = token1["name"]
            token1_name = token1_name.strip().replace(",", "(comma)")
            token1_symbol = token1["symbol"].strip().replace(",", "(comma)")
            token1_decimals = token1["decimals"]
            token1_price = pair["token1Price"]
            reserve1 = pair["reserve1"]

            createdAtBlockNumber = pair["createdAtBlockNumber"]
            createdAtTimestamp = pair["createdAtTimestamp"]

            max_createdAtTimestamp = max(max_createdAtTimestamp, int(createdAtTimestamp))

            with open(UNISWAPV2_FILE, "a") as f:
                f.write(",".join([pair_address, token0_address, token0_name, token0_symbol, str(token0_decimals), str(reserve0), str(token0_price), token1_address, token1_name, token1_symbol, str(token1_decimals), str(reserve1), str(token1_price), str(createdAtBlockNumber), str(createdAtTimestamp)]) + "\n")

        print("max_createdAtTimestamp: ", max_createdAtTimestamp)

        if len(result["data"]["pairs"]) < LIMIT:
            break



if __name__ == "__main__":
    main()
