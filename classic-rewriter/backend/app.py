"""
名著改写系统 - Flask Web 应用
提供网页界面，支持上传文本、选择风格、改写输出
"""

from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import os
import json
import uuid
from datetime import datetime
from rewriter import Rewriter
from prompts import get_prompt, get_style_list

app = Flask(__name__, 
            template_folder='../frontend',
            static_folder='../frontend')
CORS(app)

# 初始化改写器
rewriter = Rewriter()

# 存储路径
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '../data/uploads')
RESULT_FOLDER = os.path.join(os.path.dirname(__file__), '../data/results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    """首页"""
    return send_file('../frontend/index.html')

@app.route('/api/styles', methods=['GET'])
def get_styles():
    """获取所有可用的改写风格"""
    return jsonify({
        "success": True,
        "styles": get_style_list()
    })

@app.route('/api/rewrite', methods=['POST'])
def rewrite_text():
    """
    改写文本
    接收 JSON: {
        "text": "原文内容",
        "style": "风格ID",
        "title": "章节标题（可选）"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"success": False, "error": "请提供数据"}), 400
        
        text = data.get('text', '').strip()
        style_id = data.get('style', 'humorous')
        title = data.get('title', '')
        
        if not text:
            return jsonify({"success": False, "error": "请提供要改写的文本"}), 400
        
        # 获取风格 Prompt
        try:
            style_prompt = get_prompt(style_id, text)
        except ValueError as e:
            return jsonify({"success": False, "error": str(e)}), 400
        
        # 执行改写
        result = rewriter.rewrite_chapter(text, style_prompt, title)
        
        # 保存结果
        if result["success"]:
            result_id = str(uuid.uuid4())
            result_file = os.path.join(RESULT_FOLDER, f"{result_id}.json")
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "id": result_id,
                    "timestamp": datetime.now().isoformat(),
                    "original_text": text,
                    "style": style_id,
                    "result": result
                }, f, ensure_ascii=False, indent=2)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/rewrite/chapters', methods=['POST'])
def rewrite_chapters():
    """
    按章节改写
    接收 JSON: {
        "chapters": [
            {"title": "第一章", "content": "..."},
            {"title": "第二章", "content": "..."}
        ],
        "style": "风格ID"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"success": False, "error": "请提供数据"}), 400
        
        chapters = data.get('chapters', [])
        style_id = data.get('style', 'humorous')
        
        if not chapters:
            return jsonify({"success": False, "error": "请提供章节内容"}), 400
        
        # 获取风格 Prompt
        try:
            style_prompt = get_prompt(style_id, "")
        except ValueError as e:
            return jsonify({"success": False, "error": str(e)}), 400
        
        # 按章节改写
        results = []
        for i, chapter in enumerate(chapters):
            title = chapter.get('title', f'第{i+1}章')
            content = chapter.get('content', '')
            
            if content:
                result = rewriter.rewrite_chapter(
                    content, 
                    style_prompt, 
                    title
                )
                results.append(result)
        
        # 合并结果
        success_count = sum(1 for r in results if r["success"])
        
        return jsonify({
            "success": success_count > 0,
            "total_chapters": len(chapters),
            "success_chapters": success_count,
            "results": results
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """
    上传文本文件
    支持 .txt 文件
    """
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "请上传文件"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"success": False, "error": "请选择文件"}), 400
        
        # 保存文件
        filename = f"{uuid.uuid4()}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # 读取内容
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return jsonify({
            "success": True,
            "filename": file.filename,
            "content": content,
            "length": len(content)
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "model": rewriter.model
    })


if __name__ == '__main__':
    # 检查 API Key
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("警告: DASHSCOPE_API_KEY 环境变量未设置")
        print("请在 .env 文件中设置 DASHSCOPE_API_KEY")
    
    print("名著改写系统启动中...")
    print("访问 http://localhost:5000 开始使用")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
