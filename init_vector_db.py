import os
import gc
import pandas as pd
from chromadb import Client
from config import VECTOR_DB_DIR, PDF_PATH
import logging
from pdf_processor import extract_text_from_pdf, extract_articles_from_text, save_to_csv
import time

# 强制禁用 AVX512 指令集（关键优化）
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["MKL_DEBUG_CPU_TYPE"] = "5"
os.environ["ORT_CPU_THREADS"] = "2"
os.environ["OPENBLAS_CORETYPE"] = "SKYLAKE"  # 模拟 Intel 架构避免指令集冲突

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("init_vector_db.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# PDF 与 CSV 路径
PDF_PATH = "minfadian.pdf"
CSV_PATH = "minfadian.csv"
Start_time = time.time()

def init_database():
    """使用 ONNX 模型初始化向量数据库（AMD 优化版），增强日志和进度反馈"""
    logger.info("开始初始化法律助手向量数据库...")
    start_time = time.time()
    Start_time = start_time

    # 检查并创建向量数据库目录
    if not os.path.exists(VECTOR_DB_DIR):
        os.makedirs(VECTOR_DB_DIR, exist_ok=True)
        logger.info(f"创建向量数据库目录: {VECTOR_DB_DIR}")

    # 检查 CSV 文件
    if not os.path.exists(CSV_PATH):
        logger.info(f"未找到 CSV 文件，从 PDF 提取: {PDF_PATH}")
        # 从 PDF 提取条文并保存为 CSV
        try:
            text = extract_text_from_pdf(PDF_PATH)
            articles = extract_articles_from_text(text)
            save_to_csv(articles, CSV_PATH)
            logger.info(f"成功从 PDF 提取并保存 {len(articles)} 条法律条文")
        except Exception as e:
            logger.error(f"PDF 处理失败: {e}", exc_info=True)
            raise

    # 加载法律条文
    try:
        law_data = pd.read_csv(CSV_PATH)
        logger.info(f"成功加载 {len(law_data)} 条法律条文")
    except Exception as e:
        logger.error(f"CSV 加载失败: {e}", exc_info=True)
        raise

    # 初始化向量数据库
    client = Client()

    # 使用 ONNX 嵌入函数（关键修改）
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
    embedding_function = ONNXMiniLM_L6_V2()
    logger.info("使用 ONNX 嵌入函数，优化 AMD 兼容性")

    # 创建或清空集合
    try:
        collection = client.get_collection("legal_articles")
        logger.info("已存在法律条文集合，清空并重建")
        collection.delete()
    except:
        logger.info("创建新的法律条文集合")

    collection = client.create_collection(
        name="legal_articles",
        embedding_function=embedding_function
    )

    # 超小批量处理（每次仅处理 2 条），增加进度显示
    batch_size = 2
    total = len(law_data)
    logger.info(f"开始向量化处理，共 {total} 条法律条文，批次大小: {batch_size}")

    for i in range(0, total, batch_size):
        batch = law_data.iloc[i:i + batch_size]
        ids = [str(row['article']) for _, row in batch.iterrows()]
        documents = list(batch['text'])
        metadatas = [
            {"article": str(row['article']), "law": "中华人民共和国民法典"}
            for _, row in batch.iterrows()
        ]

        logger.info(f"正在处理 {i + 1}-{min(i + batch_size, total)}/{total} 条法律条文")

        # 生成嵌入向量
        embeddings = embedding_function(documents)

        # 入库
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )

        # 强制释放内存（关键优化）
        del ids, documents, metadatas, embeddings
        gc.collect()

        # 计算并显示进度百分比
        progress = min(i + batch_size, total) / total * 100
        logger.info(f"向量化进度: {progress:.2f}%")

    logger.info(f"向量数据库初始化完成，共 {collection.count()} 条法律条文，耗时: {time.time() - start_time:.2f}秒")
    return collection.count()


if __name__ == "__main__":
    try:
        count = init_database()
        print(f"✅ 向量数据库初始化成功，共 {count} 条法律条文，耗时: {time.time() - Start_time:.2f}秒")
    except Exception as e:
        print(f"❌ 向量数据库初始化失败: {e}")
        import traceback
        traceback.print_exc()