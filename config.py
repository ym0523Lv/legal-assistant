# 智谱清言API配置
ZHIPU_API_KEY = "c24f351f06e447fe9d4d7931b4db4ce0.p7ZQz1tRKjQuIkQM"  # 替换为你的API密钥
ZHIPU_API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
MODEL_NAME = "glm-4-flash"

# 向量数据库配置
VECTOR_DB_DIR = "./chroma_db"

# PDF 文件路径
PDF_PATH = "minfadian.pdf"  # 需自行准备
CSV_PATH = "minfadian.csv"               # 提取的条文 CSV
VECTOR_DB_DIR = "chroma_db"              # 向量数据库目录

DB_PATH = "legal.db"                     # 存储条文-页码映射的数据库

# AMD 处理器优化配置
import os
os.environ["OMP_NUM_THREADS"] = "4"  # 根据 CPU 核心数调整
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["MKL_DEBUG_CPU_TYPE"] = "5"  # 禁用 AVX512 避免兼容性问题
