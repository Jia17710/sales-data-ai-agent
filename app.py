import streamlit as st
import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# 导入你已经写好的agent工具（确保 commerce_agent.py 在同一个文件夹）
from commerce_agent import (
    get_macro_trend, compare_companies, company_deep_dive, analyze_correlation,
    tools, FUNCTION_MAP, clean_ai_output
)

load_dotenv()
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

st.title("行业与公司研究 Agent")
st.write("输入你的问题，AI会自动调用工具查询真实数据、分析趋势并回答")

# ============ 用 session_state 保存对话历史（网页刷新也不会丢失，直到手动清空）============
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": (
                "你是一位资深行研分析师助手，擅长宏观趋势、公司对比、财务分析与相关性研究。"
                "回答时只使用工具返回的真实数据，不要编造任何数字或单位。"
                "统计术语必须准确，相关系数不能解释为概率。"
            )
        }
    ]

# ============ 把历史对话都显示在网页上 ============
for msg in st.session_state.messages:
    if msg["role"] in ("user", "assistant") and msg.get("content"):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

# ============ 输入框（网页版的聊天输入框）============
user_input = st.chat_input("比如：拼多多和阿里巴巴的股价联动关系强不强？")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("正在思考并查询数据..."):
            response = client.chat.completions.create(
                model="openrouter/free",
                messages=st.session_state.messages,
                tools=tools
            )
            ai_message = response.choices[0].message

            if ai_message.tool_calls:
                tool_call = ai_message.tool_calls[0]
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                st.caption(f"🔧 调用工具：{function_name}({function_args})")
                result = FUNCTION_MAP[function_name](function_args)

                st.session_state.messages.append(ai_message.model_dump())
                st.session_state.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False, default=str)
                })

                final_response = client.chat.completions.create(
                    model="openrouter/free",
                    messages=st.session_state.messages
                )
                final_text = clean_ai_output(final_response.choices[0].message.content)
                st.write(final_text)
                st.session_state.messages.append({"role": "assistant", "content": final_text})
            else:
                raw_text = clean_ai_output(ai_message.content)
                st.write(raw_text)
                st.session_state.messages.append({"role": "assistant", "content": raw_text})