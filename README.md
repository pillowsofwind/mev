# Arbitrage Strategy and DL Forecasting

## Arbitrage strategy

See files under `arbitrage`.

## Pools forecasting

You may need to get a largefile `arb_data/pools.csv` manually to build the dataset!

Try the following:
~~~
    cd forecast/
    rm -r dataset/ # clear the dataset (Optional)
    python model.py # process the dataset, train the model, and evaluate it
~~~