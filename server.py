
from flask import Flask, request, jsonify, abort
import os
import json
import openai  # 使用OpenAI的Python库

app = Flask(__name__)

# 定义有效的API密钥
VALID_API_KEYS = ["11111111-2222-3333-4444-5"]

openai.api_key = os.getenv("OPENAI_API_KEY").strip()


# 加载知识库内容
def load_knowledge_base(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        knowledge_base = json.load(file)
    return knowledge_base

# 使用 LLM 进行语义搜索（适用于聊天模型）
def llm_search(query, knowledge_base):
    # 格式化知识库内容用于传递给 LLM
    context = "\n".join([f"{entry['title']}: {entry['content']}" for entry in knowledge_base])

    # 调用 GPT-3.5-turbo 或 GPT-4 模型进行聊天补全
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # 可以更换为 "gpt-4" 如果有权限使用
        messages=[
            {"role": "system", "content": "你是一个帮助回答问题的助手。"},
            {"role": "user", "content": f"基于以下信息回答问题：\n\n{context}\n\n问题：{query}"}
        ],
        max_tokens=150
    )
    
    answer = response.choices[0].message['content'].strip()
    return answer

# 验证 API 密钥的函数
def check_api_key():
    api_key = request.headers.get("x-api-key")
    if api_key not in VALID_API_KEYS:
        abort(403)  # 返回 403 Forbidden，禁止访问

# 加载知识库
knowledge_base = load_knowledge_base("knowledge_base.json")

# 定义 API 端点
@app.route("/query", methods=["GET"])
def query():
    # 验证 API 密钥
    check_api_key()
    
    # 获取查询参数并查询知识库
    query = request.args.get("query", "")
    # 调用 LLM 进行语义搜索
    answer = llm_search(query, knowledge_base)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(port=5001)
