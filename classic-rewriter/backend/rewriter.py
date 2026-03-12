"""
名著改写器 - LLM 集成模块
使用通义千问 API 进行文本改写
支持多风格、章节结构保留
"""

import dashscope
from dashscope import Generation
import os
import time
import json
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置通义千问 API
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

class Rewriter:
    def __init__(self, model="qwen-plus"):
        """
        初始化改写器
        :param model: 使用的模型名称，默认 qwen-plus
        """
        self.model = model
        self.max_retries = 3
        self.retry_delay = 2
    
    def rewrite_chapter(self, chapter_content, style_prompt, chapter_title=None):
        """
        改写单个章节
        :param chapter_content: 章节原文
        :param style_prompt: 风格 Prompt 字典 (包含 system 和 user)
        :param chapter_title: 章节标题（可选）
        :return: 改写后的章节内容
        """
        # 构建消息
        messages = [
            {
                "role": "system",
                "content": style_prompt["system"]
            },
            {
                "role": "user",
                "content": style_prompt["user"]
            }
        ]
        
        # 调用 API
        for attempt in range(self.max_retries):
            try:
                response = Generation.call(
                    model=self.model,
                    messages=messages,
                    result_format='message',
                    temperature=0.7,
                    top_p=0.8,
                    max_tokens=4000
                )
                
                if response.status_code == 200:
                    rewritten = response.output.choices[0].message.content
                    return {
                        "success": True,
                        "content": rewritten,
                        "chapter_title": chapter_title,
                        "original_length": len(chapter_content),
                        "rewritten_length": len(rewritten),
                        "model": self.model,
                        "style": style_prompt["name"]
                    }
                else:
                    print(f"API 调用失败 (尝试 {attempt+1}/{self.max_retries}): {response.message}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
            except Exception as e:
                print(f"发生错误 (尝试 {attempt+1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        
        return {
            "success": False,
            "error": "改写失败，请稍后重试",
            "chapter_title": chapter_title
        }
    
    def rewrite_full_text(self, full_text, style_prompt, preserve_structure=True):
        """
        改写完整文本
        :param full_text: 完整原文
        :param style_prompt: 风格 Prompt 字典
        :param preserve_structure: 是否保留章节结构
        :return: 改写后的完整文本
        """
        # 如果文本较短，直接整体改写
        if len(full_text) < 3000:
            return self.rewrite_chapter(full_text, style_prompt)
        
        # 如果文本较长，分章节改写
        chapters = self._split_into_chapters(full_text)
        results = []
        
        for i, chapter in enumerate(chapters):
            print(f"正在改写第 {i+1}/{len(chapters)} 章...")
            result = self.rewrite_chapter(chapter["content"], style_prompt, chapter["title"])
            results.append(result)
            
            # 避免 API 限流
            if i < len(chapters) - 1:
                time.sleep(1)
        
        # 合并结果
        return self._merge_chapters(results)
    
    def _split_into_chapters(self, text):
        """
        将文本按章节分割
        支持多种章节标记格式
        """
        import re
        
        # 常见章节标记模式
        chapter_patterns = [
            r'第[一二三四五六七八九十百千万\d]+章\s*[^\n]*',  # 第X章
            r'Chapter\s+\d+\s*[^\n]*',  # Chapter X
            r'第[一二三四五六七八九十百千万\d]+回\s*[^\n]*',  # 第X回
            r'\n\s*\d+\.\s*[^\n]*',  # 1. 标题
            r'\n\s*[一二三四五六七八九十]+、\s*[^\n]*',  # 一、标题
        ]
        
        chapters = []
        current_pos = 0
        
        for pattern in chapter_patterns:
            matches = list(re.finditer(pattern, text))
            if len(matches) > 1:  # 找到多个章节
                for i, match in enumerate(matches):
                    start = match.start()
                    if i > 0:
                        # 添加前一章内容
                        chapters.append({
                            "title": matches[i-1].group().strip(),
                            "content": text[current_pos:start].strip()
                        })
                    current_pos = start
                
                # 添加最后一章
                chapters.append({
                    "title": matches[-1].group().strip(),
                    "content": text[current_pos:].strip()
                })
                break
        
        # 如果没有找到章节标记，按段落分割
        if not chapters:
            paragraphs = text.split('\n\n')
            chunk_size = 1000  # 每个块约1000字
            current_chunk = []
            current_length = 0
            
            for para in paragraphs:
                current_chunk.append(para)
                current_length += len(para)
                
                if current_length >= chunk_size:
                    chapters.append({
                        "title": f"第{len(chapters)+1}部分",
                        "content": '\n\n'.join(current_chunk)
                    })
                    current_chunk = []
                    current_length = 0
            
            if current_chunk:
                chapters.append({
                    "title": f"第{len(chapters)+1}部分",
                    "content": '\n\n'.join(current_chunk)
                })
        
        return chapters if chapters else [{"title": "全文", "content": text}]
    
    def _merge_chapters(self, results):
        """
        合并多个章节的改写结果
        """
        merged_content = []
        success_count = 0
        
        for result in results:
            if result["success"]:
                if result.get("chapter_title"):
                    merged_content.append(f"## {result['chapter_title']}\n\n")
                merged_content.append(result["content"])
                merged_content.append("\n\n")
                success_count += 1
        
        return {
            "success": success_count > 0,
            "content": ''.join(merged_content).strip(),
            "total_chapters": len(results),
            "success_chapters": success_count
        }


def test_rewrite():
    """测试函数"""
    rewriter = Rewriter()
    
    from prompts import get_prompt
    
    test_text = "这是一个测试文本。"
    style = get_prompt("humorous", test_text)
    
    result = rewriter.rewrite_chapter(test_text, style)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    test_rewrite()
