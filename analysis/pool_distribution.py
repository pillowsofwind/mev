import pandas as pd

from collections import defaultdict, Counter

arb_pathes = ["/home/senyang/EigenPhi/arbitrage/arbitrage_all.csv", "/home/senyang/EigenPhi/arbitrage/arbitrage_no_volume.csv", "/home/senyang/EigenPhi/raw_mev/arbitrage_add.csv"]

class Pool:
    def __init__(self, pool):
        self.pool = pool
        self.count = 0
        self.revenue = []
        self.profit = []
        self.cost = []
    
    def add(self, revenue, profit, cost):
        self.count += 1
        self.revenue.append(revenue)
        self.profit.append(profit)
        self.cost.append(cost)

arb_pools = defaultdict(lambda : Pool(None))

revenue_data = defaultdict(dict)
for path in arb_pathes:
    df = pd.read_csv(path)
    columns = df.columns
    columns = [column.strip() for column in columns]
    df.columns = columns
    for blk_num, txn_hash, revenue, profit, cost in df[["blockNumber", "txHash", "revenue", "profit", "cost"]].values:
        revenue_data[int(blk_num)][txn_hash] = (revenue, profit, cost)

df = pd.read_csv("/home/senyang/EigenPhi/arbitrage_all_with_pools_and_symbols.csv")

for blk_num, txn_hash, pools in df[["blockNumber", "txHash", "pools"]].values:
    blk_num = int(blk_num)
    if blk_num not in revenue_data:
        continue
    if txn_hash not in revenue_data[blk_num]:
        continue
    revenue, profit, cost = revenue_data[blk_num][txn_hash]
    for pool in pools.split(" "):
        pool = pool.strip()
        arb_pools[pool].pool = pool
        arb_pools[pool].add(revenue, profit, cost)

# most_profit_pools = sorted(pool_list, key=lambda x: sum(x.profit), reverse=True)
# for pair in most_profit_pools[:10:]:
#     print(pair.pool, pair.count, sum(pair.profit))

# most_profit_average_pools = sorted(pool_list, key=lambda x: sum(x.profit)/len(x.profit), reverse=True)
# for pair in most_profit_average_pools[:10:]:
#     print(pair.pool, pair.count, sum(pair.profit)/len(pair.profit))

# most_profit_max_pools = sorted(pool_list, key=lambda x: max(x.profit), reverse=True)
# for pair in most_profit_max_pools[:10:]:
#     print(pair.pool, pair.count, max(pair.profit))
pool_list = list(arb_pools.values())
select_pools = set()

print("most count")
most_count_pools = sorted(pool_list, key=lambda x: x.count, reverse=True)
for pair in most_count_pools[:20:]:
    print(pair.pool, pair.count)
    select_pools.add(pair.pool)

print("most revenue")
most_revenue_pools = sorted(pool_list, key=lambda x: sum(x.revenue), reverse=True)
for pair in most_revenue_pools[:20:]:
    print(pair.pool, pair.count, sum(pair.revenue))
    select_pools.add(pair.pool)

print("most revenue average")
most_revenue_average_pools = sorted(pool_list, key=lambda x: sum(x.revenue)/len(x.revenue), reverse=True)
for pair in most_revenue_average_pools[:20:]:
    print(pair.pool, pair.count, sum(pair.revenue)/len(pair.revenue))
    select_pools.add(pair.pool)

print("most revenue max")
most_revenue_max_pools = sorted(pool_list, key=lambda x: max(x.revenue), reverse=True)
for pair in most_revenue_max_pools[:20:]:
    print(pair.pool, pair.count, max(pair.revenue))
    select_pools.add(pair.pool)

items = []
for token in select_pools:
    items.append((token, arb_pools[token].count, sum(arb_pools[token].revenue), sum(arb_pools[token].revenue)/len(arb_pools[token].revenue), max(arb_pools[token].revenue)))

df = pd.DataFrame(items, columns=["pool", "txn count", "all revenue", "average revenue per txn", "max revenue in one txn"])
df.sort_values(by="txn count", ascending=False, inplace=True)
df.to_html("pool_distribution.html", index=False)