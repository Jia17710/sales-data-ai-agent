import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ============ 1. 拉取三家公司数据 ============
tickers = ["XOM", "CVX", "NEE"]
names = {"XOM": "埃克森美孚", "CVX": "雪佛龙", "NEE": "NextEra能源"}

close_prices = {}
for ticker in tickers:
    stock = yf.Ticker(ticker)
    history = stock.history(period="5y")
    close_prices[ticker] = history["Close"]

# ============ 2. 把三家公司的收盘价合并成一张表，方便对比 ============
df_compare = pd.DataFrame(close_prices)
print(df_compare.head())
print("---")

# ============ 3. 计算每家公司的5年涨跌幅 ============
print("=== 5年累计涨跌幅对比 ===")
for ticker in tickers:
    start = df_compare[ticker].iloc[0]
    end = df_compare[ticker].iloc[-1]
    change = (end - start) / start * 100
    print(f"{names[ticker]}({ticker})：{change:.2f}%")

# ============ 4. 画对比趋势图（归一化处理，方便看谁涨得快）============
# 归一化：把每家公司的起始股价都变成100，这样才能公平比较涨幅，而不是绝对股价高低
normalized = df_compare / df_compare.iloc[0] * 100

plt.figure(figsize=(12, 6))
for ticker in tickers:
    plt.plot(normalized.index, normalized[ticker], label=names[ticker])
plt.title("能源公司股价走势对比（起始值=100）")
plt.xlabel("日期")
plt.ylabel("相对涨幅")
plt.legend()
plt.savefig("energy_comparison.png")
print("对比图已保存：energy_comparison.png")