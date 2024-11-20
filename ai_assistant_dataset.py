import json
import requests
import os

# 示例数据（标题、正文内容和类别）
raw_data = [
    {"title": "如何注册账号", "content": "用户可以通过电子邮件注册账号，点击注册按钮后输入邮箱和密码。", "category": "FAQ"},
    {"title": "支付失败解决方案", "content": "如果支付失败，请检查信用卡信息是否正确，或者联系客服。", "category": "FAQ"},
    {"title": "产品更新通知", "content": "我们很高兴宣布推出新版本，提供了更多功能和性能改进。", "category": "Announcement"},
]

# 嵌入生成函数（使用硅基流动 API）
def generate_embedding(text):
    url = "https://api.siliconflow.cn/v1/embeddings"
    payload = {
        "model": "BAAI/bge-large-zh-v1.5",
        "input": text,
        "encoding_format": "float"
    }
    headers = {
        "Authorization": "Bearer sk-lyhtlggzwngfrjnycnqchqlpkgzglkqvxibmirclxgdavsxt",  # 替换为您的实际 token
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

# 构造数据集
dataset = []
for idx, item in enumerate(raw_data):
    embedding = generate_embedding(item["content"])
    if embedding is not None:
        dataset.append({
            "id": str(idx + 1),
            "title": item["title"],
            "content": item["content"],
            "category": item["category"],
            "timestamp": "2024-11-20",
            "vector": embedding
        })

# 确保目录存在
# output_dir = "/Users/meijie/demo4/lession4/"  # 替换为实际的路径
# os.makedirs(output_dir, exist_ok=True)
# output_file = os.path.join(output_dir, "ai_assistant_dataset.json")

output_file = "ai_assistant_dataset.json"

# 保存为 JSON 文件
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)

print(f"数据集已生成并保存至：{output_file}")