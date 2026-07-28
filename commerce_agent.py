import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import wbgapi as wb
from scipy import stats
from dotenv import load_dotenv
from openai import OpenAI

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

load_dotenv()
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# ============ 研究对象 ============
COMPANIES = {
    "PDD": "拼多多(Temu母公司)",
    "BABA": "阿里巴巴(AliExpress)",
    "AMZN": "亚马逊",
    "JD": "京东"
}

# ============ 工具1：宏观趋势 - 目标市场互联网普及率 ============
def get_macro_trend():
    """查询跨境电商核心目标市场的互联网普及率趋势"""
    countries = ["USA", "GBR", "DEU", "BRA"]
    country_names = {"USA": "美国", "GBR": "英国", "DEU": "德国", "BRA": "巴西"}

    data = wb.data.DataFrame("IT.NET.USER.ZS", countries, mrv=10)
    data_clean = data.T
    data_clean = data_clean.rename(columns=country_names)
    data_clean.index = data_clean.index.str.replace("YR", "").astype(int)
    data_clean = data_clean.dropna(how="all")

    plt.figure(figsize=(10, 5))
    for country in data_clean.columns:
        plt.plot(data_clean.index, data_clean[country], label=country, marker='o')
    plt.title("主要市场互联网普及率趋势(%)")
    plt.xlabel("年份")
    plt.ylabel("互联网普及率(%)")
    plt.legend()
    plt.savefig("macro_internet_trend.png")
    plt.close()

    latest = data_clean.iloc[-1].to_dict()
    return {
        "说明": "跨境电商目标市场互联网普及率(反映线上购物潜力)",
        "最新普及率": {k: round(v, 1) for k, v in latest.items()},
        "图表": "macro_internet_trend.png"
    }

# ============ 工具2：多公司股价横向对比(归一化) ============
def compare_companies():
    """对比跨境电商相关上市公司近3年股价表现(归一化处理)"""
    tickers = list(COMPANIES.keys())
    close_prices = {}
    for ticker in tickers:
        history = yf.Ticker(ticker).history(period="3y")["Close"]
        close_prices[ticker] = history

    df_compare = pd.DataFrame(close_prices).dropna()
    normalized = df_compare / df_compare.iloc[0] * 100

    plt.figure(figsize=(10, 5))
    for ticker in tickers:
        plt.plot(normalized.index, normalized[ticker], label=COMPANIES[ticker])
    plt.title("跨境电商相关公司股价走势对比(起始值=100)")
    plt.xlabel("日期")
    plt.ylabel("相对涨幅")
    plt.legend()
    plt.savefig("company_comparison.png")
    plt.close()

    result = {}
    for ticker in tickers:
        start = df_compare[ticker].iloc[0]
        end = df_compare[ticker].iloc[-1]
        change = (end - start) / start * 100
        result[COMPANIES[ticker]] = f"{change:.2f}%"

    return {"3年累计涨跌幅": result, "图表": "company_comparison.png"}

# ============ 工具3：单公司财务深挖 ============
def company_deep_dive(ticker):
    """深入分析单个公司的财务与股价表现"""
    ticker = ticker.upper()
    stock = yf.Ticker(ticker)
    info = stock.info
    history = stock.history(period="3y")["Close"]

    start_price = history.iloc[0]
    end_price = history.iloc[-1]
    total_return = (end_price - start_price) / start_price * 100

    return {
        "公司": COMPANIES.get(ticker, ticker),
        "当前股价": round(end_price, 2),
        "3年涨跌幅": f"{total_return:.2f}%",
        "市值(美元)": info.get("marketCap", "数据缺失"),
        "市盈率PE": info.get("trailingPE", "数据缺失"),
        "营收增长率": info.get("revenueGrowth", "数据缺失"),
        "毛利率": info.get("grossMargins", "数据缺失")
    }

# ============ 工具4：相关性与回归分析 ============
def analyze_correlation(ticker1, ticker2):
    """分析两家公司股价涨跌幅的相关性与回归关系"""
    ticker1, ticker2 = ticker1.upper(), ticker2.upper()
    price1 = yf.Ticker(ticker1).history(period="3y")["Close"]
    price2 = yf.Ticker(ticker2).history(period="3y")["Close"]

    combined = pd.DataFrame({"a": price1, "b": price2}).dropna()
    combined["a_pct"] = combined["a"].pct_change()
    combined["b_pct"] = combined["b"].pct_change()
    combined = combined.dropna()

    correlation = combined["a_pct"].corr(combined["b_pct"])
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        combined["a_pct"], combined["b_pct"]
    )

    name1 = COMPANIES.get(ticker1, ticker1)
    name2 = COMPANIES.get(ticker2, ticker2)

    return {
        "对比对象": f"{name1} vs {name2}",
        "相关系数": round(correlation, 3),
        "回归斜率": round(slope, 3),
        "R平方": round(r_value ** 2, 3),
        "P值": round(p_value, 5),
        "解读": f"{name1}每变动1%，{name2}平均同向变动{round(slope,3)}%；"
                f"两者股价波动有{round(r_value**2*100,1)}%可相互解释"
    }

# ============ 工具说明书 ============
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_macro_trend",
            "description": "查询跨境电商核心目标市场(美国/英国/德国/巴西)的互联网普及率宏观趋势，反映线上购物市场潜力",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_companies",
            "description": "对比拼多多(Temu)、阿里巴巴、亚马逊、京东这几家跨境电商相关公司近3年股价表现",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "company_deep_dive",
            "description": "深入查询单个公司的财务指标，如市值、市盈率、营收增长率、毛利率",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "股票代码，如 PDD, BABA, AMZN, JD"}
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_correlation",
            "description": "分析两家公司股价涨跌幅的相关性和回归关系，量化两者的联动程度",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker1": {"type": "string", "description": "第一个股票代码"},
                    "ticker2": {"type": "string", "description": "第二个股票代码"}
                },
                "required": ["ticker1", "ticker2"]
            }
        }
    }
]
def clean_ai_output(text):
    """清洗AI输出中可能泄漏的思考过程标记"""
    markers = ["<channel|>", "</think>"]
    for marker in markers:
        if marker in text:
            text = text.split(marker)[-1].strip()
    return text
FUNCTION_MAP = {
    "get_macro_trend": lambda args: get_macro_trend(),
    "compare_companies": lambda args: compare_companies(),
    "company_deep_dive": lambda args: company_deep_dive(args["ticker"]),
    "analyze_correlation": lambda args: analyze_correlation(args["ticker1"], args["ticker2"]),
}

# ============ Agent 主流程 ============
conversation_history = [
    {
        "role": "system",
        "content": (
            "你是一位专注跨境电商行业的资深行研分析师助手。"
            "回答时只使用工具返回的真实数据，不要编造任何数字或单位。"
            "回答要简洁、专业，体现分析逻辑，适合放进行业研究报告。"
            "统计术语必须准确：相关系数(-1到1)只表示线性关系的方向和强弱，不能解释为'概率'或'百分之多少概率同向变动'；"
            "R平方才能解释为'可解释的变动比例'。避免混淆这两个概念。"
        )
    }
]

def run_agent(user_question):
    print(f"\n用户问：{user_question}")
    conversation_history.append({"role": "user", "content": user_question})

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=conversation_history,
        tools=tools
    )
    ai_message = response.choices[0].message

    if ai_message.tool_calls:
        tool_call = ai_message.tool_calls[0]
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)

        print(f"[AI决定调用] {function_name}({function_args})")
        result = FUNCTION_MAP[function_name](function_args)

        conversation_history.append(ai_message)
        conversation_history.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result, ensure_ascii=False, default=str)
        })

        final_response = client.chat.completions.create(
            model="openrouter/free",
            messages=conversation_history
        )
        final_text = final_response.choices[0].message.content
        final_text = clean_ai_output(final_text)
        conversation_history.append({"role": "assistant", "content": final_text})
        print("回答：", final_text)
    else:
        raw_text = clean_ai_output(ai_message.content)
        conversation_history.append({"role": "assistant", "content": raw_text})
        print("回答：", raw_text)


if __name__ == "__main__":
    run_agent("跨境电商目标市场的互联网普及率趋势怎么样？")
    run_agent("帮我对比一下这几家跨境电商相关公司的股价表现")
    run_agent("详细分析一下拼多多这家公司")
    run_agent("拼多多和阿里巴巴的股价联动关系强不强？")