import yfinance as yf
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
xom = yf.Ticker("XOM")
history = xom.history(period="5y")

# 只保留收盘价这一列，方便分析
close_price = history["Close"]

# ============ 1. 计算移动平均线（判断趋势用，行研/交易分析常用）============
history["MA30"] = close_price.rolling(window=30).mean()   # 30天移动平均
history["MA90"] = close_price.rolling(window=90).mean()   # 90天移动平均

# ============ 2. 计算区间涨跌幅（类似"年初至今涨了多少"这种表述）============
start_price = close_price.iloc[0]
end_price = close_price.iloc[-1]
total_return = (end_price - start_price) / start_price * 100

print(f"5年前股价：{start_price:.2f}")
print(f"最新股价：{end_price:.2f}")
print(f"累计涨跌幅：{total_return:.2f}%")

# ============ 3. 画趋势图（股价 + 两条移动平均线）============
plt.figure(figsize=(12, 6))
plt.plot(history.index, history["Close"], label="收盘价", alpha=0.5)
plt.plot(history.index, history["MA30"], label="30日均线")
plt.plot(history.index, history["MA90"], label="90日均线")
plt.title("埃克森美孚(XOM) 股价趋势与移动平均线")
plt.xlabel("日期")
plt.ylabel("股价(美元)")
plt.legend()
plt.savefig("xom_trend.png")
print("趋势图已保存：xom_trend.png")