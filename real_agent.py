import os
import json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

df = pd.read_excel("sales_data.xlsx")

# ============ 工具函数 ============
def query_sales_by_region(region):
    result = df[df["地区"] == region]["销售额"].sum()
    return int(result)

def query_max_region():
    grouped = df.groupby("地区")["销售额"].sum()
    return {"地区": grouped.idxmax(), "销售额": int(grouped.max())}
def add_numbers(a, b):
    return a + b
def create_sales_chart():
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    grouped = df.groupby("地区")["销售额"].sum()
    
    plt.figure(figsize=(8, 5))
    plt.bar(grouped.index, grouped.values)
    plt.title("各地区销售额汇总")
    plt.xlabel("地区")
    plt.ylabel("销售额")
    plt.savefig("sales_chart.png")
    plt.close()
    
    return "图表已生成，保存为 sales_chart.png"
tools = [
    {
        "type": "function",
        "function": {
            "name": "query_sales_by_region",
            "description": "查询某个地区的总销售额",
            "parameters": {
                "type": "object",
                "properties": {
                    "region": {"type": "string", "description": "地区名称，比如 华东、华北、华南"}
                },
                "required": ["region"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_max_region",
            "description": "找出销售额最高的地区",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "计算两个数字的和",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_sales_chart",
            "description": "生成一张各地区销售额汇总的柱状图，保存为图片文件",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]


# ============ 关键改动：把对话历史存在这个列表里，全局保留 ============
conversation_history = [
    {
        "role": "system",
        "content": "你是一个数据分析助手。回答时只使用工具返回的原始数字，不要自己添加或编造任何单位（如万元、亿元），除非数据中明确提供了单位。回答要简洁准确。"
    }
]

def run_agent(user_question):
    print(f"用户问：{user_question}")
    
    # 把这句新问题加入历史记录
    conversation_history.append({"role": "user", "content": user_question})
    
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=conversation_history,   # 注意：这里发送的是完整历史，不只是这一句话
        tools=tools
    )
    
    ai_message = response.choices[0].message
    
    if ai_message.tool_calls:
        tool_call = ai_message.tool_calls[0]
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        print(f"[AI决定调用] {function_name}({function_args})")
        
        if function_name == "query_sales_by_region":
            result = query_sales_by_region(function_args["region"])
        elif function_name == "query_max_region":
            result = query_max_region()
        elif function_name == "add_numbers":
            result = add_numbers(function_args["a"], function_args["b"])
        elif function_name == "create_sales_chart":
            result = create_sales_chart()
        conversation_history.append(ai_message)
        conversation_history.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result, ensure_ascii=False)
        })
        
        final_response = client.chat.completions.create(
            model="openrouter/free",
            messages=conversation_history
        )
        final_text = final_response.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": final_text})
        print("回答：", final_text)
    else:
        conversation_history.append({"role": "assistant", "content": ai_message.content})
        print("回答：", ai_message.content)

# ============ 测试连续对话 ============
run_agent("帮我生成一张各地区销售额的图表")