"""
第3步：构建向量数据库
------------------------------------
用途：把结构化 JSON 数据（第2步的产出）灌入本地向量数据库，供第4步检索使用。

依赖安装（终端运行）：
    pip install chromadb sentence-transformers --break-system-packages

embedding 模型用 bge-large-zh-v1.5：本地运行、免费、中文语义检索效果好，
第一次运行会自动下载模型（约1.3GB），之后离线可用。
"""

import json
import chromadb
from sentence_transformers import SentenceTransformer

# ---------- 1. 初始化 ----------

# 本地持久化存储，数据存在这个文件夹里，重启程序数据不会丢
client = chromadb.PersistentClient(path="./exhibition_vector_db")

# 建一个集合（collection），相当于一张表
collection = client.get_or_create_collection(name="exhibition_talks")

# 加载中文 embedding 模型（本地运行，免费）
embed_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")


# ---------- 2. 读取第2步的结构化数据 ----------

with open(r"E:\Ritone\展会语音agent\结构化数据_Json\展会对话结构化数据.json", "r", encoding="utf-8") as f:
    exhibitions = json.load(f)


# ---------- 3. 拼接"要做向量化的文本" + 提取 metadata ----------

def build_embedding_text(segment: dict) -> str:
    """
    决定哪些字段参与语义检索。
    用提炼过的摘要+标签+价值点，而不是原始转写全文——
    原始转写口语化噪声多，会拉低检索准确率。
    """
    parts = [
        " ".join(segment.get("topic_tags", [])),
        segment.get("summary_counterpart", ""),
        segment.get("value_points", ""),
    ]
    return " ".join(p for p in parts if p)


def build_metadata(exhibition_name: str, date: str, segment: dict) -> dict:
    """
    决定哪些字段用于精确过滤（不参与语义相似度计算）。
    chromadb 的 metadata 只支持字符串/数字/布尔，
    所以 counterpart_profile 这种嵌套字典要拍平成单独字段。
    """
    profile = segment.get("counterpart_profile", {})
    return {
        "exhibition": exhibition_name,
        "date": date,
        "segment_id": str(segment.get("segment_id", "")),
        "region": profile.get("region", "未提及"),
        "business_type": profile.get("business_type", "未提及"),
        "confidence": segment.get("confidence", ""),
    }


# ---------- 4. 批量写入向量数据库 ----------

ids, documents, metadatas = [], [], []

for exhibition in exhibitions:
    ex_name = exhibition["exhibition"]
    ex_date = exhibition["date"]
    for seg in exhibition["segments"]:
        uid = f"{ex_name}_{seg['segment_id']}"  # 唯一ID，避免重复写入冲突
        ids.append(uid)
        documents.append(build_embedding_text(seg))
        metadatas.append(build_metadata(ex_name, ex_date, seg))

# 批量生成向量（一次性算完比逐条算快很多）
embeddings = embed_model.encode(documents, normalize_embeddings=True).tolist()

collection.upsert(  # upsert = 有则更新、无则新增，方便你重复跑脚本处理新展会
    ids=ids,
    embeddings=embeddings,
    documents=documents,
    metadatas=metadatas,
)

print(f"已写入 {len(ids)} 条对话记录到向量数据库 ./exhibition_vector_db")
