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
# 常用世界银行指标代码对照表(中文名 -> 世界银行官方指标代码)
# 需要更多指标时，可以去 https://data.worldbank.org 搜索后往这里添加
MACRO_INDICATORS = {
    "互联网普及率": "IT.NET.USER.ZS",
    "GDP总量": "NY.GDP.MKTP.CD",
    "人均GDP": "NY.GDP.PCAP.CD",
    "人口总量": "SP.POP.TOTL",
    "通货膨胀率": "FP.CPI.TOTL.ZG",
    "失业率": "SL.UEM.TOTL.ZS",
    "出口总额": "NE.EXP.GNFS.CD",
    "人均能源消费量": "EG.USE.PCAP.KG.OE",
}
# 常用国家中文名 -> ISO三位代码对照表(应对AI可能传中文名而非代码的情况)
COUNTRY_CODE_MAP = {
    "中国": "CHN", "美国": "USA", "英国": "GBR", "德国": "DEU",
    "法国": "FRA", "日本": "JPN", "印度": "IND", "巴西": "BRA",
    "俄罗斯": "RUS", "韩国": "KOR", "加拿大": "CAN", "澳大利亚": "AUS",
}

def normalize_country_code(country):
    """把国家标识安全转换成ISO三位代码：已经是代码就直接用，是中文名就查表转换"""
    country = country.strip()
    if len(country) == 3 and country.isalpha():
        return country.upper()   # 看起来已经是ISO代码，直接使用
    return COUNTRY_CODE_MAP.get(country, None)   # 尝试从中文名转换，转不了返回None
# ============ 工具1：宏观趋势 - 目标市场互联网普及率 ============
def get_macro_trend(indicator_name, countries):
    indicator_code = MACRO_INDICATORS.get(indicator_name)
    if not indicator_code:
        return {
            "错误": f"暂不支持指标'{indicator_name}'",
            "可选指标": list(MACRO_INDICATORS.keys())
        }

    # 安全转换国家代码，过滤掉转换失败的
    normalized_countries = []
    for c in countries:
        code = normalize_country_code(c)
        if code:
            normalized_countries.append(code)

    if not normalized_countries:
        return {"错误": f"无法识别提供的国家：{countries}，请使用ISO三位代码，如 USA、CHN"}

    data = wb.data.DataFrame(indicator_code, normalized_countries, mrv=10)
    data_clean = data.T
    data_clean.index = data_clean.index.str.replace("YR", "").astype(int)
    data_clean = data_clean.dropna(how="all")

    plt.figure(figsize=(10, 5))
    for country in data_clean.columns:
        plt.plot(data_clean.index, data_clean[country], label=country, marker='o')
    plt.title(f"{indicator_name}趋势对比")
    plt.xlabel("年份")
    plt.ylabel(indicator_name)
    plt.legend()
    plt.savefig("macro_trend_chart.png")
    plt.close()

    latest = data_clean.iloc[-1].to_dict()
    return {
        "指标": indicator_name,
        "最新数值": {k: round(v, 2) if pd.notna(v) else None for k, v in latest.items()},
        "图表": "macro_trend_chart.png"
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
            "description": "查询任意国家的宏观经济/社会指标趋势，如GDP、人口、互联网普及率、通胀率等",
            "parameters": {
                "type": "object",
                "properties": {
                    "indicator_name": {
                        "type": "string",
                        "description": "指标名称，可选：互联网普及率、GDP总量、人均GDP、人口总量、通货膨胀率、失业率、出口总额"
                    },
                    "countries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "指标名称，可选：互联网普及率、GDP总量、人均GDP、人口总量、通货膨胀率、失业率、出口总额、人均能源消费量"
                    }
                },
                "required": ["indicator_name", "countries"]
            }
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
    if text is None:
        return "抱歉，模型这次没有返回有效内容，请重新提问一次。"
    
    # 检测模型是否输出了伪造的工具调用格式(而不是走真正的function calling)
    # 这种情况下内容本身没有参考价值，直接提示用户重试
    suspicious_patterns = ["<TOOLCALL>", "toolbench_rapidapi_key"]
    if any(pattern in text for pattern in suspicious_patterns):
        return "⚠️ 本次模型返回格式异常(可能是免费模型不稳定导致)，请重新提问一次，或换一种问法。"
    
    markers = ["<channel|>", "</think>"]
    for marker in markers:
        if marker in text:
            text = text.split(marker)[-1].strip()
    return text
FUNCTION_MAP = {
    "get_macro_trend": lambda args: get_macro_trend(args["indicator_name"], args["countries"]),
    "compare_companies": lambda args: compare_companies(),
    "company_deep_dive": lambda args: company_deep_dive(args["ticker"]),
    "analyze_correlation": lambda args: analyze_correlation(args["ticker1"], args["ticker2"]),
}

# ============ Agent 主流程 ============
conversation_history = [
    {
        "role": "system",
        "content": (
            "你是一位资深行研分析师助手，擅长宏观趋势、公司对比、财务分析与相关性研究。"
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