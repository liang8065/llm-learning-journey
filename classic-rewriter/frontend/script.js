/**
 * 名著改写器 - 前端逻辑
 * 处理用户交互、API 调用、结果显示
 */

// API 基础 URL
const API_BASE = '';

// 风格列表
let styles = [];

// 选中的风格
let selectedStyles = [];

// DOM 元素
const inputText = document.getElementById('input-text');
const charCount = document.getElementById('char-count');
const styleGrid = document.getElementById('style-grid');
const btnRewrite = document.getElementById('btn-rewrite');
const btnClear = document.getElementById('btn-clear');
const btnPaste = document.getElementById('btn-paste');
const btnUpload = document.getElementById('btn-upload');
const btnCopy = document.getElementById('btn-copy');
const fileInput = document.getElementById('file-input');
const loading = document.getElementById('loading');
const resultSection = document.getElementById('result-section');
const resultContent = document.getElementById('result-content');
const resultInfo = document.getElementById('result-info');

// 初始化
document.addEventListener('DOMContentLoaded', () => {
    loadStyles();
    setupEventListeners();
});

// 加载风格列表
async function loadStyles() {
    try {
        const response = await fetch(`${API_BASE}/api/styles`);
        const data = await response.json();
        
        if (data.success) {
            styles = data.styles;
            renderStyles();
        }
    } catch (error) {
        console.error('加载风格失败:', error);
        styleGrid.innerHTML = '<p style="color: red;">加载风格失败，请刷新页面重试</p>';
    }
}

// 渲染风格选项
function renderStyles() {
    styleGrid.innerHTML = styles.map(style => `
        <div class="style-item" data-style="${style.id}">
            <span class="emoji">${style.emoji}</span>
            <div>
                <div class="style-name">${style.name}</div>
                <div class="style-desc">${style.description}</div>
            </div>
        </div>
    `).join('');
    
    // 绑定点击事件
    document.querySelectorAll('.style-item').forEach(item => {
        item.addEventListener('click', () => toggleStyle(item));
    });
}

// 切换风格选择
function toggleStyle(item) {
    const styleId = item.dataset.style;
    
    if (item.classList.contains('selected')) {
        item.classList.remove('selected');
        selectedStyles = selectedStyles.filter(s => s !== styleId);
    } else {
        item.classList.add('selected');
        selectedStyles.push(styleId);
    }
}

// 设置事件监听
function setupEventListeners() {
    // 字数统计
    inputText.addEventListener('input', () => {
        charCount.textContent = inputText.value.length;
    });
    
    // 粘贴按钮
    btnPaste.addEventListener('click', async () => {
        try {
            const text = await navigator.clipboard.readText();
            inputText.value = text;
            charCount.textContent = text.length;
        } catch (error) {
            alert('无法读取剪贴板，请手动粘贴');
        }
    });
    
    // 上传按钮
    btnUpload.addEventListener('click', () => {
        fileInput.click();
    });
    
    // 文件选择
    fileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch(`${API_BASE}/api/upload`, {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            
            if (data.success) {
                inputText.value = data.content;
                charCount.textContent = data.content.length;
            } else {
                alert('上传失败: ' + data.error);
            }
        } catch (error) {
            alert('上传失败: ' + error.message);
        }
    });
    
    // 改写按钮
    btnRewrite.addEventListener('click', rewrite);
    
    // 清空按钮
    btnClear.addEventListener('click', () => {
        inputText.value = '';
        charCount.textContent = '0';
        resultSection.style.display = 'none';
        resultContent.innerHTML = '';
        document.querySelectorAll('.style-item').forEach(item => {
            item.classList.remove('selected');
        });
        selectedStyles = [];
    });
    
    // 复制按钮
    btnCopy.addEventListener('click', () => {
        const text = resultContent.textContent;
        navigator.clipboard.writeText(text).then(() => {
            const originalText = btnCopy.textContent;
            btnCopy.textContent = '✅ 已复制';
            setTimeout(() => {
                btnCopy.textContent = originalText;
            }, 2000);
        });
    });
}

// 执行改写
async function rewrite() {
    const text = inputText.value.trim();
    
    if (!text) {
        alert('请输入要改写的文本');
        return;
    }
    
    if (selectedStyles.length === 0) {
        alert('请至少选择一种改写风格');
        return;
    }
    
    // 显示加载状态
    loading.style.display = 'block';
    resultSection.style.display = 'none';
    btnRewrite.disabled = true;
    
    try {
        const results = [];
        
        // 对每种选中的风格进行改写
        for (const style of selectedStyles) {
            const response = await fetch(`${API_BASE}/api/rewrite`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    text: text,
                    style: style
                })
            });
            
            const data = await response.json();
            results.push({
                style: style,
                styleName: styles.find(s => s.id === style)?.name || style,
                emoji: styles.find(s => s.id === style)?.emoji || '📝',
                ...data
            });
        }
        
        // 显示结果
        displayResults(results, text.length);
        
    } catch (error) {
        alert('改写失败: ' + error.message);
    } finally {
        loading.style.display = 'none';
        btnRewrite.disabled = false;
    }
}

// 显示结果
function displayResults(results, originalLength) {
    resultSection.style.display = 'block';
    
    const successCount = results.filter(r => r.success).length;
    resultInfo.textContent = `完成 ${successCount}/${results.length} 种风格改写 | 原文 ${originalLength} 字`;
    
    let html = '';
    
    for (const result of results) {
        if (result.success) {
            html += `
                <div class="result-item">
                    <h3>${result.emoji} ${result.styleName}风格</h3>
                    <p style="color: #666; font-size: 0.9rem; margin-bottom: 10px;">
                        改写长度: ${result.rewritten_length} 字 | 模型: ${result.model}
                    </p>
                    <div class="result-text">${escapeHtml(result.content)}</div>
                </div>
                <hr style="margin: 20px 0; border: none; border-top: 1px solid #e0e0e0;">
            `;
        } else {
            html += `
                <div class="result-item error">
                    <h3>${result.emoji} ${result.styleName}风格</h3>
                    <p style="color: #dc3545;">❌ 改写失败: ${result.error || '未知错误'}</p>
                </div>
            `;
        }
    }
    
    resultContent.innerHTML = html;
}

// HTML 转义
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
