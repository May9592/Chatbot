import json
import requests
from pinecone import Pinecone, ServerlessSpec
import os

# 配置 Pinecone 的 API Key
PC_API_KEY = os.getenv("PC_API_KEY").strip() # 替换为您的 Pinecone API Key
pc = Pinecone(api_key=PC_API_KEY)  # 替换为您的 Pinecone API Key

API_TOKEN =  os.getenv("API_TOKEN").strip()  # 替换为您的 SiliconFlow API Key

try:
    print("验证 API Key 中...")
    print("索引列表：", pc.list_indexes().names())
except Exception as e:
    print(f"API Key 验证失败：{e}")
    exit()

# 如果索引已存在，先删除
index_name = "ai-assistant-index"
if index_name in pc.list_indexes().names():
    pc.delete_index(index_name)
    print(f"索引 {index_name} 已被删除。")


# 连接到 Pinecone 索引
index_name = "ai-assistant-index"  # 索引名称
vector_dimension = 1024

try:
    # 检查是否存在索引
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=vector_dimension,  # 嵌入向量的维度
            metric="cosine",  # 使用余弦相似度
            spec=ServerlessSpec(cloud="aws", region="us-east-1")  # 使用有效的云和区域配置
        )
    index = pc.Index(index_name)
except Exception as e:
    print(f"初始化 Pinecone 索引失败：{e}")
    exit()

# 加载本地数据集
def load_dataset(file_path):

    # if index_name in pc.list_indexes().names():
    #     pc.delete_index(index_name)
    # print(f"索引 {index_name} 已被删除。")
    
    """加载本地 JSON 数据集并上传到 Pinecone。"""
    with open(file_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        print(f"嵌入向量维度：{len(dataset[0]['vector'])}")
    vectors = [
        {
            "id": item["id"],
            "values": item["vector"],  # 嵌入向量
            "metadata": {
                "title": item["title"],
                "content": item["content"],
                "category": item["category"],
                "timestamp": item["timestamp"]
            }
        }
        for item in dataset
    ]
    index.upsert(vectors=vectors)
    print(f"数据集已成功加载到 Pinecone 索引 '{index_name}'。")

# 使用 SiliconFlow 生成嵌入向量
def generate_embedding_with_siliconflow(text):
    url = "https://api.siliconflow.cn/v1/embeddings"
    payload = {
        "model": "BAAI/bge-large-zh-v1.5",
        "input": text,
        "encoding_format": "float"
    }
    headers = {
        "Authorization": "Bearer " + API_TOKEN,  # 替换为您的实际 token
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # 检查是否有 HTTP 错误
        data = response.json()
        return data.get("data", [])[0].get("embedding")  # 返回嵌入向量
    except Exception as e:
        print(f"生成嵌入失败：{e}")
        return None

# 用户输入处理函数
def get_llm_response(user_input):
    """
    根据用户输入生成最终的 AI 响应，包括：
    1. 生成输入文本的嵌入向量。
    2. 检索向量数据库中的最相关内容。
    3. 使用 SiliconFlow API 基于上下文生成最终回答。
    """
    # Step 1: 生成用户输入的嵌入向量
    try:
        query_vector = generate_embedding_with_siliconflow(user_input)
    except Exception as e:
        return f"生成嵌入失败：{e}"

    # Step 2: 在 Pinecone 中检索最相关的内容
    try:
        search_results = index.query(
            vector=query_vector,  # 查询向量
            top_k=3,  # 返回最相关的 3 条内容
            include_metadata=True  # 包含元数据（如标题和内容）
        )
    except Exception as e:
        return f"向量数据库检索失败：{e}"

    # Step 3: 提取检索结果并构造上下文
    if "matches" not in search_results or not search_results["matches"]:
        return "很抱歉，我没有找到相关的内容。"

    # 构造 SiliconFlow API 的上下文
    context = "以下是与用户问题相关的内容：\\n"
    for match in search_results["matches"]:
        metadata = match["metadata"]
        context += f"- 标题: {metadata['title']}\\n  内容: {metadata['content']}\\n\\n"

    # Step 4: 使用 SiliconFlow API 生成最终回答
    try:
        api_url = "https://api.siliconflow.cn/v1/chat/completions"
        headers = {
            "Authorization": "Bearer " + API_TOKEN,  # 替换为您的 SiliconFlow API Key
            "Content-Type": "application/json"
        }
        payload = {
            "model": "Qwen/Qwen2.5-7B-Instruct",  # 或替换为您使用的模型
            "messages": [
                {"role": "system", "content": "你是一个智能助手，根据提供的上下文回答用户问题。"},
                {"role": "user", "content": f"用户问题：{user_input}\\n\\n{context}"}
            ]
        }
        response = requests.post(api_url, headers=headers, json=payload)
        response_data = response.json()
        if response.status_code == 200:
            return response_data["choices"][0]["message"]["content"]
        else:
            return f"SiliconFlow API 返回错误：{response_data.get('error', '未知错误')}"
    except Exception as e:
        return f"生成 SiliconFlow API 响应失败：{e}"

# 主运行逻辑
if __name__ == "__main__":
    # 加载本地数据集（JSON 文件路径）
    dataset_path = "ai_assistant_dataset.json"  # 替换为您的数据集路径
    try:
        load_dataset(dataset_path)
    except Exception as e:
        print(f"加载数据集失败：{e}")
        exit()

    print("欢迎使用智能 AI 助手！")
    while True:
        # 接受用户输入
        user_input = input("请输入您的问题（输入 '退出' 以结束）：\\n")
        if user_input.lower() in ["退出", "exit", "quit"]:  # 支持多种退出命令
            print("感谢您的使用，再见！")
            break
        # 获取 AI 助手的响应
        response = get_llm_response(user_input)
        print(response)