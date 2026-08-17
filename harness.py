"""
第4步：Harness —— 把"检索+生成"串成可复用的调用
------------------------------------
依赖安装：
    pip install chromadb sentence-transformers openai --break-system-packages

需要设置环境变量 DEEPSEEK_API_KEY（你自己的 API Key）。
DeepSeek 用的是 OpenAI 兼容格式，所以用 openai 这个库调用，
只是把 base_url 指向 DeepSeek 的服务器地址。
"""

import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os

# ---------- 1. 初始化（跟 build_vector_db.py 用同一份数据库和模型）----------

client = chromadb.PersistentClient(path="./exhibition_vector_db")
collection = client.get_collection(name="exhibition_talks")
embed_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

llm = OpenAI(
    api_key="sk-aa6bf7d0aa5b401d9184cd9de9e696bd",  # 自动读取环境变量
    base_url="https://api.deepseek.com",
)


# ---------- 2. 检索层：语义检索 + metadata 精确过滤 ----------

def retrieve(query: str, top_k: int = 8, where: dict | None = None) -> list[dict]:
    """
    query: 用户的自然语言问题
    where: 可选的精确过滤条件，例如 {"exhibition": "食物展览"}
           chromadb 的 where 语法，多条件要用 $and 包裹，例如：
           {"$and": [{"exhibition": "食物展览"}, {"confidence": "高"}]}
    """
    query_vec = embed_model.encode([query], normalize_embeddings=True).tolist()

    results = collection.query(
        query_embeddings=query_vec,
        n_results=top_k,
        where=where,  # 不需要过滤就传 None
    )

    # 把 chromadb 返回的结构，整理成更好用的列表
    hits = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        hits.append({"content": doc, "metadata": meta})
    return hits


# ---------- 3. 任务A：查询式问答（精确、不编造）----------

def answer_query(question: str, where: dict | None = None) -> str:
    hits = retrieve(question, top_k=8, where=where)

    context = "\n\n".join(
        f"[{h['metadata']['exhibition']} | {h['metadata']['region']} | "
        f"{h['metadata']['business_type']}]\n{h['content']}"
        for h in hits
    )

    prompt = f"""以下是从展会对话记录里检索到的相关片段，请基于这些内容回答问题。
只使用给出的信息回答，不要编造信息里没有的内容；如果检索结果中没有能回答
问题的信息，直接说明"没有找到相关记录"。

检索到的片段：
{context}

问题：{question}
"""

    response = llm.chat.completions.create(
        model="deepseek-chat",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


# ---------- 4. 任务B：归纳生成（提炼观点写文章）----------

def generate_summary(topic_or_exhibition: str, where: dict | None = None) -> str:
    # 归纳任务通常需要更多上下文覆盖面，top_k 给大一点
    hits = retrieve(topic_or_exhibition, top_k=20, where=where)

    context = "\n\n".join(
        f"[{h['metadata']['exhibition']} | {h['metadata']['region']} | "
        f"{h['metadata']['business_type']}]\n{h['content']}"
        for h in hits
    )

    prompt = f"""以下是展会对话记录的若干片段，请围绕"{topic_or_exhibition}"这个主题，
提炼出2-3个有洞察力的观点，写成一段适合发公众号的文字（300字左右，
中文，有观点、不是流水账罗列）。可以引用具体的事实细节增强说服力，
但不要编造记录中没有的信息。

对话记录：
{context}
"""

    response = llm.chat.completions.create(
        model="deepseek-chat",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


# ---------- 5. 使用示例 ----------

if __name__ == "__main__":
    # 任务A示例：精确查询
    print(answer_query("哪些人聊到了出口或出海相关的话题？"))

    print("\n" + "=" * 50 + "\n")

    # 任务B示例：归纳生成，可以用 where 限定某场展会
    print(generate_summary("中国市场对国际参展商的吸引力", where={"exhibition": "食物展览"}))
