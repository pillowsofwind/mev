import pandas as pd

from collections import defaultdict, Counter

label_path = "/home/senyang/EigenPhi/blockchair_erc-20_tokens_latest.tsv"
with open(label_path, "r") as f:
    lines = f.readlines()[1::]
    lines = [line.strip() for line in lines]
    lines = [line.split("\t") for line in lines]
    address_to_symbol = dict(zip([line[1] for line in lines], [line[4] for line in lines]))

# label_df = pd.read_csv(label_path, sep="\t")

# address_to_symbol = dict(zip(label_df["address"], label_df["symbol"]))

arb_pathes = ["/home/senyang/EigenPhi/arbitrage/arbitrage_all.csv", "/home/senyang/EigenPhi/arbitrage/arbitrage_no_volume.csv", "/home/senyang/EigenPhi/raw_mev/arbitrage_add.csv"]

class TokenPair:
    def __init__(self, token_pair):
        self.token_pair = token_pair
        self.count = 0
        self.revenue = []
        self.profit = []
        self.cost = []
    
    def add(self, revenue, profit, cost):
        self.count += 1
        self.revenue.append(revenue)
        self.profit.append(profit)
        self.cost.append(cost)

class Token:
    def __init__(self, token):
        self.token = token
        self.count = 0
        self.revenue = []
        self.profit = []
        self.cost = []
    
    def add(self, revenue, profit, cost):
        self.count += 1
        self.revenue.append(revenue)
        self.profit.append(profit)
        self.cost.append(cost)

tokens = defaultdict(lambda : Token(None))

for path in arb_pathes:
    df = pd.read_csv(path)
    columns = df.columns
    columns = [column.strip() for column in columns]
    df.columns = columns
    for token_pair, revenue, profit, cost in df[["tokens", "revenue", "profit", "cost"]].values:
        pair = token_pair.split(" ")
        pair_symbols = []
        
        for token_addr in pair:
            token_addr = token_addr.strip()[2::]
            if token_addr not in address_to_symbol:
                pair_symbols.append(token_addr)
            else:
                pair_symbols.append(address_to_symbol[token_addr])

        # pair_name = "/".join(pair_symbols)
        for symbol in pair_symbols:
            tokens[symbol].token = symbol
            tokens[symbol].add(revenue, profit, cost)

        # for token_symbol in pair_symbols:
        #     tokens[token_symbol] += 1

token_list = tokens.values()

# most_profit_tokens = sorted(token_list, key=lambda x: sum(x.profit), reverse=True)
# for pair in most_profit_tokens[:10:]:
#     print(pair.token, pair.count, sum(pair.profit))

# most_profit_average_tokens = sorted(token_list, key=lambda x: sum(x.profit)/len(x.profit), reverse=True)
# for pair in most_profit_average_tokens[:10:]:
#     print(pair.token, pair.count, sum(pair.profit)/len(pair.profit))

# most_profit_max_tokens = sorted(token_list, key=lambda x: max(x.profit), reverse=True)
# for pair in most_profit_max_tokens[:10:]:
#     print(pair.token, pair.count, max(pair.profit))
select_tokens = set()

print("most count")
most_count_tokens = sorted(token_list, key=lambda x: x.count, reverse=True)
for pair in most_count_tokens[:20:]:
    print(pair.token, pair.count)
    select_tokens.add(pair.token)

print("most revenue")
most_revenue_tokens = sorted(token_list, key=lambda x: sum(x.revenue), reverse=True)
for pair in most_revenue_tokens[:20:]:
    print(pair.token, pair.count, sum(pair.revenue))
    select_tokens.add(pair.token)

print("most revenue average")
most_revenue_average_tokens = sorted(token_list, key=lambda x: sum(x.revenue)/len(x.revenue), reverse=True)
for pair in most_revenue_average_tokens[:20:]:
    print(pair.token, pair.count, sum(pair.revenue)/len(pair.revenue))
    select_tokens.add(pair.token)

print("most revenue max")
most_revenue_max_tokens = sorted(token_list, key=lambda x: max(x.revenue), reverse=True)
for pair in most_revenue_max_tokens[:20:]:
    print(pair.token, pair.count, max(pair.revenue))
    select_tokens.add(pair.token)

items = []
for token in select_tokens:
    items.append((token, tokens[token].count, sum(tokens[token].revenue), sum(tokens[token].revenue)/len(tokens[token].revenue), max(tokens[token].revenue)))

df = pd.DataFrame(items, columns=["token", "txn count", "all revenue", "average revenue per txn", "max revenue in one txn"])
df.sort_values(by="txn count", ascending=False, inplace=True)
df.to_html("token_distribution.html", index=False)