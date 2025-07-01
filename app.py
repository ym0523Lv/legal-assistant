import os
import time
import re
import json
from flask import Flask, request, jsonify, render_template, send_file, make_response
from flask_cors import CORS
import logging
import requests
from chromadb import Client
from sentence_transformers import SentenceTransformer
from config import ZHIPU_API_KEY, ZHIPU_API_URL, MODEL_NAME, PDF_PATH  # 替换为你的配置
from prompt_templates import LAWYER_PROMPT  # 替换为你的提示词

# 性能优化环境变量（按需调整）
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # 允许所有跨域请求

# 全局变量（向量数据库 + 模型）
client = None
collection = None
embedding_model = None
init_time = 0


def load_embedding_model():
    """加载 SentenceTransformer 模型（带异常处理）"""
    global embedding_model, init_time
    start_time = time.time()
    try:
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("✅ 嵌入模型加载成功")
        init_time = time.time() - start_time
    except Exception as e:
        logger.error(f"❌ 嵌入模型加载失败: {e}")
        raise


def init_vector_db():
    """初始化 ChromaDB 向量数据库（带重试逻辑）"""
    global client, collection
    start_time = time.time()
    try:
        load_embedding_model()  # 先加载模型

        client = Client()  # 初始化 ChromaDB 客户端
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

        # 尝试加载或创建集合
        try:
            collection = client.get_collection(
                name="legal_articles",
                embedding_function=embedding_function
            )
            logger.info(f"✅ 加载现有向量集合（{collection.count()} 条数据）")
        except Exception as e:
            logger.warning(f"⚠️ 获取集合失败，创建新集合: {e}")
            collection = client.create_collection(
                name="legal_articles",
                embedding_function=embedding_function
            )
            logger.info("✅ 新法律条文向量集合创建成功")

        logger.info(f"⏱️ 向量数据库初始化耗时: {time.time() - start_time:.2f} 秒")
    except Exception as e:
        logger.error(f"❌ 向量数据库初始化失败: {e}")
        collection = None
        raise


def call_llm(messages, max_retries=3):
    """调用大模型（带重试 + 超时）"""
    headers = {
        "Authorization": f"Bearer {ZHIPU_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 1024
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(
                ZHIPU_API_URL,
                headers=headers,
                json=payload,
                timeout=(10, 30)  # 连接超时 10s，读取超时 30s
            )
            response.raise_for_status()  # 检查 HTTP 状态码
            result = response.json()

            # 校验响应格式
            if 'choices' in result and result['choices']:
                return result['choices'][0]['message']['content']
            else:
                logger.error(f"❌ 大模型响应异常: {result}")
                return "抱歉，大模型返回格式异常，请重试"

        except requests.Timeout:
            logger.warning(f"⚠️ 大模型调用超时（第 {attempt + 1}/{max_retries} 次重试）")
        except requests.ConnectionError:
            logger.warning(f"⚠️ 网络连接异常（第 {attempt + 1}/{max_retries} 次重试）")
        except Exception as e:
            logger.error(f"❌ 大模型调用失败: {e}")

    return "抱歉，大模型服务暂时不可用，请稍后再试"


# ------------------------- 路由配置 ------------------------- #
@app.route('/')
def index():
    """首页（初始化向量数据库 + 传递状态）"""
    try:
        if collection is None:
            init_vector_db()  # 延迟初始化
        return render_template(
            'index.html',
            init_time=init_time,
            init_failed=False,
            pdf_path=PDF_PATH  # 传递 PDF 路径给前端
        )
    except Exception as e:
        logger.error(f"❌ 首页初始化失败: {e}")
        return render_template(
            'index.html',
            init_time=0,
            init_failed=True,
            error_msg=str(e)
        )


@app.route('/chat', methods=['POST'])
def chat():
    """聊天接口（法律条文高亮 + 右侧联动）"""
    start_time = time.time()
    try:
        if collection is None:
            init_vector_db()  # 确保向量库已加载

        data = request.json
        user_question = data.get('message', '').strip()
        if not user_question:
            return jsonify({
                "status": "error",
                "reply": "请输入有效问题",
                "context": "",
                "error": "empty_question",
                "process_time": 0
            })

        # 1. 向量数据库检索相关法律条文
        query_embedding = embedding_model.encode([user_question]).tolist()[0]
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=["documents", "metadatas"]
        )

        # 2. 构建上下文（法律条文）
        context = "\n\n".join([
            f"《民法典》第{meta['article']}条：{doc}"
            for doc, meta in zip(results['documents'][0], results['metadatas'][0])
        ])

        # 3. 调用大模型生成回答
        messages = [
            {"role": "system", "content": LAWYER_PROMPT},
            {"role": "user", "content": f"{context}\n\n用户问题：{user_question}"}
        ]
        llm_reply = call_llm(messages)

        # 4. 格式化回答（法律条文高亮 + 可点击）
        formatted_reply = re.sub(
            r'《([^》]+)》第(\d+)条',
            r'<span class="law-ref" data-law="\1" data-article="\2">《\1》第\2条</span>',
            llm_reply
        )

        # 5. 返回结果（右侧联动所需数据）
        return jsonify({
            "status": "success",
            "reply": formatted_reply,
            "context": context,
            "process_time": time.time() - start_time,
            "error": ""
        })

    except Exception as e:
        logger.error(f"❌ 聊天请求异常: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "reply": f"处理失败: {str(e)}",
            "context": "",
            "process_time": time.time() - start_time,
            "error": str(e)
        })


@app.route('/get_article', methods=['GET'])
def get_article():
    """法律条文详情接口（右侧详情框内容）"""
    try:
        if collection is None:
            init_vector_db()

        article = request.args.get('article')  # 接收条文编号
        if not article:
            return make_response(
                jsonify({"status": "error", "content": "缺少 article 参数"}),
                400
            )

        # 向量数据库查询条文内容
        results = collection.query(
            query_texts=[f"第{article}条"],
            n_results=1,
            where={"article": article},
            include=["documents", "metadatas"]
        )

        if results['documents'] and results['documents'][0]:
            return jsonify({
                "status": "success",
                "content": results['documents'][0][0],
                "article": article,
                "metadata": results['metadatas'][0][0]
            })
        else:
            return make_response(
                jsonify({"status": "error", "content": "未找到对应条文"}),
                404
            )
    except Exception as e:
        logger.error(f"❌ 条文查询异常: {e}")
        return make_response(
            jsonify({"status": "error", "content": "查询失败"}),
            500
        )


@app.route('/get_pdf')
def get_pdf():
    """PDF 预览（支持在线查看）"""
    try:
        if not os.path.exists(PDF_PATH):
            return make_response(
                jsonify({"status": "error", "error": "PDF 文件不存在"}),
                404
            )

        # 在线预览 PDF（设置 Content-Disposition）
        response = make_response(send_file(
            PDF_PATH,
            mimetype='application/pdf'
        ))
        response.headers.set(
            'Content-Disposition',
            'inline',  # 在线预览（ attachment 为强制下载）
            filename=os.path.basename(PDF_PATH)
        )
        return response
    except Exception as e:
        logger.error(f"❌ PDF 下载失败: {e}")
        return make_response(
            jsonify({"status": "error", "error": "PDF 访问异常"}),
            500
        )


@app.route('/health')
def health_check():
    """健康检查（向量库状态 + 模型状态）"""
    return jsonify({
        "status": "ok" if collection is not None else "initializing",
        "collection_status": "ready" if collection is not None else "not_initialized",
        "vector_count": collection.count() if collection is not None else 0,
        "timestamp": time.time(),
        "model_status": "loaded" if embedding_model is not None else "unloaded"
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)