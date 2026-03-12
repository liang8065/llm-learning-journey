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
const resultStats = document.getElementById('result-stats');

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
        } else {
            throw new Error(data.error || '加载风格失败');
        }
    } catch (error) {
        console.error('加载风格失败:', error);
        styleGrid.innerHTML = '<p style="color: var(--error); grid-column: 1/-1; text-align: center;">加载风格失败，请刷新页面重试</p>';
    }
}

// 渲染风格选项
function renderStyles() {
    styleGrid.innerHTML = styles.map(style => `
        <div class="style-item" data-style="${style.id}">
            <span class="style-emoji">${style.emoji}</span>
            <div class="style-info">
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
        const count = inputText.value.length;
        charCount.textContent = `${count} 字`;
    });
    
    // 粘贴按钮
    btnPaste.addEventListener('click', async () => {
        try {
            const text = await navigator.clipboard.readText();
            inputText.value = text;
            charCount.textContent = `${text.length} 字`;
            showNotification('已从剪贴板粘贴', 'success');
        } catch (error) {
            showNotification('无法读取剪贴板，请手动粘贴', 'warning');
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
                charCount.textContent = `${data.content.length} 字`;
                showNotification(`文件 "${file.filename}" 上传成功`, 'success');
            } else {
                showNotification('上传失败: ' + data.error, 'error');
            }
        } catch (error) {
            showNotification('上传失败: ' + error.message, 'error');
        }
        
        // 重置文件输入
        fileInput.value = '';
    });
    
    // 改写按钮
    btnRewrite.addEventListener('click', rewrite);
    
    // 清空按钮
    btnClear.addEventListener('click', () => {
        inputText.value = '';
        charCount.textContent = '0 字';
        resultSection.classList.remove('active');
        resultContent.innerHTML = '';
        resultStats.innerHTML = '';
        document.querySelectorAll('.style-item').forEach(item => {
            item.classList.remove('selected');
        });
        selectedStyles = [];
        showNotification('已清空', 'info');
    });
    
    // 复制按钮
    btnCopy.addEventListener('click', () => {
        const text = resultContent.textContent;
        navigator.clipboard.writeText(text).then(() => {
            const originalText = btnCopy.innerHTML;
            btnCopy.innerHTML = '<span class="btn-icon">✅</span> 已复制';
            setTimeout(() => {
                btnCopy.innerHTML = originalText;
            }, 2000);
        }).catch(() => {
            showNotification('复制失败，请手动选择复制', 'error');
        });
    });
}

// 执行改写
async function rewrite() {
    const text = inputText.value.trim();
    
    if (!text) {
        showNotification('请输入要改写的文本', 'warning');
        return;
    }
    
    if (selectedStyles.length === 0) {
        showNotification('请至少选择一种改写风格', 'warning');
        return;
    }
    
    // 显示加载状态
    loading.classList.add('active');
    resultSection.classList.remove('active');
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
        showNotification('改写完成！', 'success');
        
    } catch (error) {
        showNotification('改写失败: ' + error.message, 'error');
    } finally {
        loading.classList.remove('active');
        btnRewrite.disabled = false;
    }
}

// 显示结果
function displayResults(results, originalLength) {
    resultSection.classList.add('active');
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    
    const successCount = results.filter(r => r.success).length;
    resultInfo.textContent = `完成 ${successCount}/${results.length} 种风格改写`;
    
    // 统计信息
    let totalRewritten = 0;
    results.forEach(r => {
        if (r.success && r.rewritten_length) {
            totalRewritten += r.rewritten_length;
        }
    });
    
    resultStats.innerHTML = `
        <div class="stat-item">
            <span>📝</span>
            <span>原文 <span class="stat-value">${originalLength}</span> 字</span>
        </div>
        <div class="stat-item">
            <span>✨</span>
            <span>改写 <span class="stat-value">${totalRewritten}</span> 字</span>
        </div>
        <div class="stat-item">
            <span>🎯</span>
            <span>成功 <span class="stat-value">${successCount}</span> 种</span>
        </div>
    `;
    
    let html = '';
    
    for (const result of results) {
        if (result.success) {
            html += `
                <div class="result-item fade-in">
                    <div class="result-item-header">
                        <span style="font-size: 1.3rem;">${result.emoji}</span>
                        <span class="result-item-title">${result.styleName}风格</span>
                        <div class="result-meta">
                            <span>📝 ${result.rewritten_length || '-'} 字</span>
                            <span>🤖 ${result.model || 'qwen-plus'}</span>
                        </div>
                    </div>
                    <div class="result-text">${escapeHtml(result.content)}</div>
                </div>
            `;
        } else {
            html += `
                <div class="result-item error fade-in">
                    <div class="result-item-header">
                        <span style="font-size: 1.3rem;">${result.emoji}</span>
                        <span class="result-item-title">${result.styleName}风格</span>
                    </div>
                    <p class="result-error">❌ 改写失败: ${result.error || '未知错误'}</p>
                </div>
            `;
        }
    }
    
    resultContent.innerHTML = html;
}

// 显示通知
function showNotification(message, type = 'info') {
    // 创建通知元素
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 25px;
        border-radius: 12px;
        background: white;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        z-index: 9999;
        display: flex;
        align-items: center;
        gap: 10px;
        font-weight: 500;
        animation: slideIn 0.3s ease;
        max-width: 300px;
    `;
    
    // 根据类型设置颜色
    const colors = {
        success: '#4CAF50',
        error: '#F44336',
        warning: '#FF9800',
        info: '#8B4513'
    };
    
    const icons = {
        success: '✅',
        error: '❌',
        warning: '⚠️',
        info: 'ℹ️'
    };
    
    notification.style.borderLeft = `4px solid ${colors[type]}`;
    notification.innerHTML = `<span>${icons[type]}</span><span>${message}</span>`;
    
    document.body.appendChild(notification);
    
    // 添加动画样式
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        @keyframes slideOut {
            from { transform: translateX(0); opacity: 1; }
            to { transform: translateX(100%); opacity: 0; }
        }
    `;
    document.head.appendChild(style);
    
    // 3秒后移除
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => {
            notification.remove();
        }, 300);
    }, 3000);
}

// HTML 转义
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
