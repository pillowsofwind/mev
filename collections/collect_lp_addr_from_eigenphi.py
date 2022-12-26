import os
import requests
import pandas as pd


EIGENPHI_API_URL = "https://eigenphi.io/api/v2/arbitrage/stat/lp/hotLp/?chain=ethereum&pageNum={}&pageSize={}&period=30&sortBy=volume"
DST_FILE = os.path.join(os.path.dirname(__file__), "../data/lp_addr_list.csv")


def get_lp_addr_from_eigenphi(page_num, page_size):
    url = EIGENPHI_API_URL.format(page_num, page_size)
    response = requests.get(url)
    assert(response.status_code == 200)
    data = response.json()
    return data["result"]["data"]


def main():
    page_size = 100
    lp_addr_list = []
    for page_num in range(1, 11):
        lp_addr_list += get_lp_addr_from_eigenphi(page_num, page_size)
    
    items = []
    for item in lp_addr_list:
        new_item = {}
        new_item["lpAddress"] = item["lpAddress"]
        new_item["lpName"] = item["lpName"]
        if "protocol" in item and "name" in item["protocol"]:
            new_item["protocolName"] = item["protocol"]["name"]
        else:
            new_item["protocolName"] = ""

        token_index = 0
        for token in item["tokens"]:
            new_item["token{}_address".format(token_index)] = token["address"]
            new_item["token{}_symbol".format(token_index)] = token["symbol"]
            new_item["token{}_decimals".format(token_index)] = token["decimals"]
            token_index += 1

        items.append(new_item)

    df = pd.DataFrame(items)

    df.to_csv(DST_FILE, index=False)


if __name__ == "__main__":
    main()
