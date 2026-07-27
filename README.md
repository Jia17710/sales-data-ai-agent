# 销售数据分析 AI Agent

一个基于大语言模型 Function Calling 机制的数据分析助手，能够理解自然语言问题，自动调用工具查询 Excel 销售数据、进行计算、生成可视化图表。

## 功能特性

- 🔍 **自然语言查询**：用中文提问即可查询销售数据，无需写 SQL 或代码
- 💬 **多轮对话记忆**：支持连续追问，AI 能记住上下文（如"那华东呢？"）
- 🛠️ **工具调用（Function Calling）**：AI 自主判断何时调用工具，包括查询、计算、生成图表
- 📊 **数据可视化**：自动生成销售数据柱状图
- 🎯 **减少幻觉**：通过 System Prompt 约束模型输出，避免编造数据单位等信息

## 技术栈

- Python 3.14
- pandas（数据处理）
- matplotlib（数据可视化）
- OpenRouter API（大语言模型调用，兼容 OpenAI SDK）
- Function Calling / Tool Use（AI 工具调用机制）

## 项目结构