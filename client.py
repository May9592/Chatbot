
import requests
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

# 服务器地址和端口
server_url = "http://localhost:5001/query"
headers = {
    "x-api-key": "11111111-2222-3333-4444-5"  # 确保与服务器端一致
}

print("开始对话（输入'退出'以结束对话）")

while True:
    # 获取用户输入的问题
    user_input = input("你: ")
    
    # 检查是否退出对话
    if user_input.lower() == "退出":
        print("结束对话")
        break

    # 向服务器发送请求
    response = requests.get(server_url, params={"query": user_input}, headers=headers)

    # 检查响应状态并输出回答
    if response.status_code == 200:
        try:
            results = response.json()  # 尝试将响应转换为 JSON
            if "answer" in results:
                answer = results["answer"]
                print("助手:", answer)
            else:
                print("Unexpected response format:", results)
        except ValueError:
            print("响应不是JSON格式:", response.text)
    else:
        print("访问被拒绝或无效的API密钥")