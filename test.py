# test_init.py
from chromadb import Client
from chromadb.utils import embedding_functions

# 创建简单示例数据
example_data = [
    {"article": 1, "text": "为了保护民事主体的合法权益，调整民事关系..."},
    {"article": 2, "text": "民法调整平等主体的自然人、法人和非法人组织之间的人身关系和财产关系。"}
]

# 初始化向量数据库
client = Client()
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# 创建集合
collection = client.create_collection(
    name="legal_articles",
    embedding_function=embedding_function
)

# 添加示例数据
for item in example_data:
    collection.add(
        ids=str(item["article"]),
        documents=item["text"],
        metadatas={"article": str(item["article"]), "law": "中华人民共和国民法典"}
    )

print(f"成功导入 {len(example_data)} 条测试数据")