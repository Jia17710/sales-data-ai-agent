import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

oil = yf.Ticker("CL=F").history(period="5y")["Close"]
xom = yf.Ticker("XOM").history(period="5y")["Close"]

combined = pd.DataFrame({"原油价格": oil, "股价": xom})
combined = combined.dropna()

# ============ 新增：计算每日涨跌幅（收益率），而不是用原始价格 ============
combined["原油涨跌幅"] = combined["原油价格"].pct_change()
combined["股价涨跌幅"] = combined["股价"].pct_change()

combined = combined.dropna()   # pct_change第一行会产生NaN，去掉

print(combined[["原油涨跌幅", "股价涨跌幅"]].head())
print("---")

# ============ 用涨跌幅重新计算相关系数 ============
correlation = combined["原油涨跌幅"].corr(combined["股价涨跌幅"])
print(f"原油涨跌幅与XOM股价涨跌幅的相关系数：{correlation:.3f}")
# ============ 线性回归：用原油涨跌幅预测股价涨跌幅 ============
x = combined["原油涨跌幅"]
y = combined["股价涨跌幅"]

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

print(f"回归系数(斜率)：{slope:.3f}")
print(f"截距：{intercept:.4f}")
print(f"R平方值：{r_value**2:.3f}")
print(f"P值：{p_value:.5f}")

print(f"\n解读：原油每上涨1%，XOM股价平均变动 {slope:.3f}%")