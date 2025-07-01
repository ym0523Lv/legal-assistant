import PyPDF2
import re
from collections import defaultdict
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path):
    """从 PDF 中提取文本（优化 AMD 兼容性）"""
    logger.info(f"开始提取 PDF 文本: {pdf_path}")

    try:
        # 尝试直接提取文本
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text
                if i % 50 == 0 and i > 0:
                    logger.info(f"已提取 {i} 页文本")
            logger.info(f"成功提取 {len(reader.pages)} 页文本")
            return text

    except Exception as e:
        logger.error(f"直接提取文本失败: {e}")

        # 尝试分段提取（处理大文件）
        try:
            return _extract_text_by_chunks(pdf_path)
        except Exception as e2:
            logger.error(f"分段提取文本失败: {e2}")
            raise Exception("PDF 文本提取失败，请检查文件格式")


def _extract_text_by_chunks(pdf_path, chunk_size=50):
    """分段处理大 PDF 文件"""
    from PyPDF2 import PdfReader, PdfWriter

    logger.info(f"开始分段提取 PDF 文本，每段 {chunk_size} 页")
    full_text = ""

    with open(pdf_path, 'rb') as f:
        reader = PdfReader(f)
        total_pages = len(reader.pages)

        for i in range(0, total_pages, chunk_size):
            # 创建临时 PDF
            writer = PdfWriter()
            for j in range(i, min(i + chunk_size, total_pages)):
                writer.add_page(reader.pages[j])

            # 保存临时文件
            temp_path = f"temp_{i}.pdf"
            with open(temp_path, 'wb') as temp_f:
                writer.write(temp_f)

            # 提取文本
            try:
                with open(temp_path, 'rb') as temp_f:
                    temp_reader = PyPDF2.PdfReader(temp_f)
                    for page in temp_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            full_text += page_text
            finally:
                # 删除临时文件
                os.remove(temp_path)

            logger.info(f"已处理 {min(i + chunk_size, total_pages)}/{total_pages} 页")

    return full_text


def extract_articles_from_text(text):
    """从提取的文本中识别法律条文"""
    logger.info("开始解析法律条文...")

    articles = {}
    current_article = None
    current_content = []

    # 识别条文标题（如"第一千零四十条"）
    article_pattern = r'第([一二三四五六七八九十百千]+)条'
    lines = text.split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 检查是否为新条文
        match = re.match(article_pattern, line)
        if match:
            # 保存上一条文
            if current_article and current_content:
                articles[current_article] = ' '.join(current_content).strip()

            # 开始新条文
            current_article = match.group(1)
            current_content = [line[match.end():].strip()]
        else:
            # 继续当前条文内容
            if current_article:
                current_content.append(line)

    # 保存最后一条文
    if current_article and current_content:
        articles[current_article] = ' '.join(current_content).strip()

    logger.info(f"成功解析 {len(articles)} 条法律条文")
    return articles


def chinese_number_to_arabic(chinese_num):
    """将中文数字转换为阿拉伯数字"""
    chinese_nums = {
        '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
        '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
        '百': 100, '千': 1000
    }

    result = 0
    temp = 0

    for char in chinese_num:
        if char in ['十', '百', '千']:
            if temp == 0:
                temp = 1
            temp *= chinese_nums[char]
            result += temp
            temp = 0
        else:
            temp = chinese_nums[char]

    # 处理如"十"、"十一"等特殊情况
    if temp > 0:
        result += temp

    return result


def save_to_csv(articles, csv_path):
    """将提取的条文保存为 CSV"""
    import csv

    logger.info(f"将法律条文保存到 {csv_path}...")

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['article', 'text'])

        for article, content in articles.items():
            try:
                # 转换中文数字为阿拉伯数字
                arabic_num = chinese_number_to_arabic(article)
                writer.writerow([arabic_num, content])
            except Exception as e:
                logger.warning(f"无法转换条文编号: {article} - {e}")
                writer.writerow([article, content])  # 保留原始格式


# 使用示例
if __name__ == "__main__":
    pdf_path = "minfadian.pdf"
    csv_path = "minfadian.csv"

    text = extract_text_from_pdf(pdf_path)
    articles = extract_articles_from_text(text)
    save_to_csv(articles, csv_path)

    print(f"成功提取 {len(articles)} 条法律条文并保存到 {csv_path}")