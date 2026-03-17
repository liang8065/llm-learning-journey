# LLaMA Factory 深度技术解析

> 🔍 对 [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LlamaFactory) 的全面技术分析
> 📅 分析日期: 2026-03-17
> ⭐ 项目数据: 40k+ Stars | 100+ 模型支持 | ACL 2024 论文

## 目录

- [1. 项目概览](#1-项目概览)
- [2. 架构设计](#2-架构设计)
- [3. 核心模块详解](#3-核心模块详解)
- [4. 支持的训练方法](#4-支持的训练方法)
- [5. 支持的模型](#5-支持的模型)
- [6. 数据处理流水线](#6-数据处理流水线)
- [7. 优化技术](#7-优化技术)
- [8. 配置参数全解](#8-配置参数全解)
- [9. 使用指南](#9-使用指南)
- [10. 技术亮点与设计哲学](#10-技术亮点与设计哲学)
- [11. 与其他框架对比](#11-与其他框架对比)
- [12. 总结与建议](#12-总结与建议)

---

## 1. 项目概览

### 1.1 基本信息

| 属性 | 详情 |
|------|------|
| **项目名称** | LLaMA Factory (llamafactory) |
| **作者** | hiyouga (北京航空航天大学) |
| **语言** | Python (>=3.11) |
| **许可证** | Apache-2.0 |
| **构建系统** | Hatchling |
| **学术论文** | ACL 2024 — "LLaMA Board: A Zero-code Platform for Fine-tuning Large Language Models" |
| **CLI 入口** | `llamafactory-cli` 或 `lmf` |

### 1.2 一句话总结

**零代码微调 100+ 大语言模型的统一平台**，通过 CLI 和 Web UI (Gradio) 提供从预训练到 RLHF 的全流程支持。

### 1.3 核心依赖

```
torch >= 2.4.0
transformers >= 4.51.0, <= 5.2.0
peft >= 0.18.0 (参数高效微调)
trl >= 0.18.0 (强化学习训练)
datasets >= 2.16.0
accelerate >= 1.3.0 (分布式训练)
gradio >= 4.38.0 (Web UI)
```

---

## 2. 架构设计

### 2.1 顶层架构

```
src/
├── api.py              # OpenAI 风格 API 服务入口
├── train.py            # 训练入口
├── webui.py            # Gradio Web UI 入口
└── llamafactory/       # 核心库
    ├── api/            # API 服务模块 (FastAPI)
    ├── chat/           # 对话推理模块
    ├── cli.py          # CLI 入口
    ├── data/           # 数据处理流水线
    ├── eval/           # 评估模块
    ├── extras/         # 工具函数和常量
    ├── hparams/        # 超参数定义
    ├── launcher.py     # 分布式启动器
    ├── model/          # 模型加载与适配
    ├── train/          # 训练器 (SFT/DPO/KTO/PPO/RM)
    └── webui/          # Gradio UI 组件
```

### 2.2 分层架构图

```
┌─────────────────────────────────────────────────────┐
│                   用户接口层                          │
│   CLI (cli.py) │ WebUI (webui.py) │ API (api.py)    │
├─────────────────────────────────────────────────────┤
│                   业务逻辑层                          │
│   launcher.py (任务分发与分布式协调)                    │
├──────────┬──────────┬──────────┬────────────────────┤
│  Model   │   Data   │  Train   │   Chat / Eval      │
│  Module  │  Module  │  Module  │   Module           │
├──────────┴──────────┴──────────┴────────────────────┤
│                   基础设施层                          │
│   hparams │ extras (logging/misc/packages)           │
├─────────────────────────────────────────────────────┤
│                   外部依赖层                          │
│   transformers │ peft │ trl │ accelerate │ datasets  │
└─────────────────────────────────────────────────────┘
```

### 2.3 设计原则

1. **模块解耦**: 模型、数据、训练三大模块独立，可灵活组合
2. **配置驱动**: 通过 YAML 配置文件 + dataclass 参数定义，实现零代码操作
3. **多入口**: CLI / WebUI / API 三种使用方式共享同一核心逻辑
4. **可扩展**: 支持自定义数据集、模板、模型适配器

---

## 3. 核心模块详解

### 3.1 Model 模块 (`src/llamafactory/model/`)

#### 模型加载流程 (`loader.py`)

```python
# 核心加载流程
def load_model(tokenizer, model_args, finetuning_args, is_trainable, add_valuehead):
    1. 加载配置 (AutoConfig)
    2. 补丁配置 (patch_config) - 适配不同模型的特殊配置
    3. 应用 Liger Kernel (可选加速)
    4. 选择加载策略:
       - KTransformers (CPU+GPU 混合推理)
       - Unsloth (LoRA 训练加速)
       - 标准 HuggingFace 加载
    5. 模型类型识别:
       - AutoModelForImageTextToText (多模态图文)
       - AutoModelForSeq2SeqLM (音频文本)
       - AutoModelForTextToWaveform (Qwen Omni)
       - AutoModelForCausalLM (标准因果语言模型)
    6. 应用补丁 (patch_model)
    7. 初始化适配器 (init_adapter)
```

#### 适配器初始化 (`adapter.py`)

支持的适配器类型：
- **LoRA / QLoRA**: 通过 PEFT 库实现
- **Freeze Tuning**: 冻结底层，只训练顶层
- **Full Fine-tuning**: 全参数微调
- **OFT / OFTv2**: 正交微调 (2025年新增)
- **GaLore / APOLLO**: 梯度低秩投影优化

#### 模型补丁 (`patcher.py`)

关键补丁操作：
- `patch_config`: 修改模型配置（RoPE 缩放、注意力函数等）
- `patch_tokenizer`: 修复 tokenizer 兼容性问题
- `patch_processor`: 处理多模态处理器
- `patch_model`: 模型级修改（梯度检查点、特殊模块处理）
- `patch_valuehead_model`: PPO 奖励模型头

### 3.2 Data 模块 (`src/llamafactory/data/`)

#### 数据加载流程 (`loader.py`)

```
数据源支持:
├── HuggingFace Hub (hf_hub)
├── ModelScope Hub (ms_hub)
├── Modelers Hub (om_hub)
├── 本地脚本 (script)
├── 本地文件 (file) - 支持 json/jsonl/csv/parquet/arrow
└── 云端文件 (cloud_file)

数据处理流程:
原始数据 → align_dataset (格式对齐) → DatasetProcessor (tokenization) → 训练数据
```

#### 数据集处理器 (`processor.py`)

| 处理器 | 用途 | 训练阶段 |
|--------|------|----------|
| `PretrainDatasetProcessor` | 预训练数据处理 | pt |
| `SupervisedDatasetProcessor` | 指令微调数据处理 | sft |
| `PackedSupervisedDatasetProcessor` | 序列打包的 SFT | sft (packing=True) |
| `PairwiseDatasetProcessor` | 偏好数据对处理 | rm |
| `FeedbackDatasetProcessor` | KTO 反馈数据 | kto |
| `UnsupervisedDatasetProcessor` | 无监督数据 | ppo/eval |

#### 模板系统

支持的对话模板覆盖主流模型：
- LLaMA 系列: llama2, llama3, llama4
- Qwen 系列: qwen, qwen2, qwen2_vl, qwen3
- ChatGLM 系列: chatglm2, chatglm3, glm4
- 其他: mistral, gemma, phi, deepseek, yi, baichuan 等

### 3.3 Train 模块 (`src/llamafactory/train/`)

#### 训练器目录结构

```
train/
├── callbacks.py      # 训练回调 (日志、保存、评估)
├── fp8_utils.py      # FP8 混合精度工具
├── pt/               # 预训练 (Pre-training)
├── sft/              # 监督微调 (Supervised Fine-Tuning)
├── rm/               # 奖励模型 (Reward Modeling)
├── ppo/              # PPO 强化学习
├── dpo/              # DPO 直接偏好优化
├── kto/              # KTO (Kahneman-Tversky Optimization)
├── orpo/             # ORPO (Odds Ratio Preference Optimization)
├── grpo/             # GRPO (Group Relative Policy Optimization)
└── extras/           # 训练辅助工具
```

#### 训练入口 (`src/train.py`)

```python
# 极简入口
from llamafactory.train import run_exp
run_exp()  # 从配置文件读取所有参数并启动训练
```

---

## 4. 支持的训练方法

### 4.1 训练阶段概览

| 阶段 | 方法 | 说明 |
|------|------|------|
| **pt** | 预训练 | 继续预训练，学习新知识 |
| **sft** | 监督微调 | 指令跟随、对话能力 |
| **rm** | 奖励模型训练 | 为 RLHF 训练奖励模型 |
| **ppo** | PPO 强化学习 | 基于人类反馈的强化学习 |
| **dpo** | DPO 直接偏好优化 | 无需奖励模型的偏好对齐 |
| **kto** | KTO 优化 | Kahneman-Tversky 偏好优化 |
| **orpo** | ORPO 优化 | 比率偏好优化 |
| **grpo** | GRPO 优化 | 组相对策略优化 |

### 4.2 LoRA 变体

| 方法 | 参数 | 说明 |
|------|------|------|
| **LoRA** | `finetuning_type=lora` | 标准低秩适配 |
| **QLoRA** | `quantization_bit=4/8` + lora | 量化 + LoRA |
| **rsLoRA** | `use_rslora=True` | 秩稳定缩放 LoRA |
| **DoRA** | `use_dora=True` | 权重分解 LoRA |
| **LoRA+** | `loraplus_lr_ratio=20` | 学习率差异化 |
| **PiSSA** | `pissa_init=True` | 奇异值分解初始化 |
| **OFT** | `finetuning_type=oft` | 正交微调 (2025新增) |
| **OFTv2** | — | 改进版正交微调 |

### 4.3 全参数微调变体

| 方法 | 说明 |
|------|------|
| **Full Tuning** | 全参数微调 |
| **Freeze Tuning** | 冻结底层参数 |
| **GaLore** | 梯度低秩投影 (省内存) |
| **APOLLO** | 优化的梯度投影方法 |
| **BAdam** | 块级 Adam 优化 |
| **Adam-mini** | 减少优化器状态的 Adam |
| **Muon** | 基于矩阵正交化的优化器 |

---

## 5. 支持的模型

### 5.1 模型族支持 (100+ 模型)

| 模型族 | 代表模型 | 模态 |
|--------|----------|------|
| **LLaMA** | LLaMA 2/3/4 | 文本/多模态 |
| **Qwen** | Qwen 2.5/3, Qwen2.5-VL, Qwen3-VL | 文本/视觉/音频 |
| **DeepSeek** | DeepSeek-R1, DeepSeek-V3 | 文本 |
| **Gemma** | Gemma 2/3 | 文本/视觉 |
| **GLM** | GLM-4, GLM-4.1V, GLM-Z1 | 文本/视觉 |
| **Mistral** | Mistral, Mixtral-MoE | 文本 |
| **Phi** | Phi-3/4 | 文本/视觉 |
| **InternLM** | InternLM 3, InternVL 3 | 文本/视觉 |
| **MiniCPM** | MiniCPM-V/o-2.6 | 视觉/音频 |
| **Yi** | Yi, Yi-VL | 文本/视觉 |
| **其他** | Baichuan, ChatGLM, Command-R, OLMo, etc. | 文本 |

### 5.2 Day-0/Day-1 支持承诺

项目承诺对新模型的快速适配：
- **Day 0 支持**: Qwen3, Qwen2.5-VL, Gemma 3, GLM-4.1V, InternLM 3, MiniCPM-o-2.6
- **Day 1 支持**: Llama 3/4, GLM-4, Mistral Small, PaliGemma2

### 5.3 多模态能力

| 模态 | 能力 | 代表模型 |
|------|------|----------|
| 图像理解 | 视觉问答、图像描述 | LLaVA, Qwen-VL, InternVL |
| 视频识别 | 视频理解 | Qwen2-VL |
| 音频理解 | 语音识别、音频问答 | Qwen2-Audio, MiniCPM-o |
| 视觉定位 | 目标检测/定位 | — |
| 图文生成 | ImageTextToText | — |

---

## 6. 数据处理流水线

### 6.1 数据格式

LLaMA Factory 定义了标准化的数据格式：

#### SFT 数据格式 (alpaca 格式)

```json
{
  "instruction": "请解释什么是机器学习",
  "input": "",
  "output": "机器学习是人工智能的一个分支..."
}
```

#### SFT 数据格式 (sharegpt 格式)

```json
{
  "conversations": [
    {"from": "human", "value": "你好"},
    {"from": "gpt", "value": "你好！有什么可以帮助你的？"},
    {"from": "human", "value": "解释一下深度学习"},
    {"from": "gpt", "value": "深度学习是..."}
  ]
}
```

#### 偏好数据格式 (DPO/RM)

```json
{
  "instruction": "写一首关于春天的诗",
  "input": "",
  "chosen": "春天来了，万物复苏...",
  "rejected": "春。好。"
}
```

### 6.2 数据集配置 (`dataset_info.json`)

```json
{
  "my_dataset": {
    "file_name": "my_data.json",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content"
    }
  }
}
```

### 6.3 数据处理流程

```
1. 解析数据集配置 → get_dataset_list()
2. 加载原始数据 → _load_single_dataset()
   - 支持从 HF/MS/Om Hub 加载
   - 支持本地文件 (json/jsonl/csv/parquet)
   - 支持流式加载 (streaming=True)
3. 格式对齐 → align_dataset()
   - 将不同格式统一为标准列
4. 数据集合并 → merge_dataset()
   - concat: 拼接
   - interleave_under/over: 交错采样
5. Tokenization → DatasetProcessor.process()
   - 应用对话模板
   - 截断/填充到 cutoff_len
   - 生成 labels 和 attention_mask
6. 可选处理:
   - 序列打包 (packing=True)
   - 流式加载 (streaming=True)
   - 缓存到磁盘 (tokenized_path)
```

---

## 7. 优化技术

### 7.1 训练加速

| 技术 | 参数 | 说明 |
|------|------|------|
| **FlashAttention-2** | `flash_attn=auto/fa2` | 高效注意力计算 |
| **Unsloth** | `use_unsloth=True` | LoRA 训练 2x 加速 |
| **Liger Kernel** | `enable_liger_kernel=True` | 内核级优化 |
| **KTransformers** | `use_kt=True` | CPU+GPU 混合推理 |
| **序列打包** | `packing=True` | 提高 GPU 利用率 |
| **Neat Packing** | `neat_packing=True` | 无交叉注意力的打包 |
| **梯度检查点** | 默认启用 | 用计算换内存 |
| **FP8 训练** | `fp8=True` | Hopper 架构 GPU 加速 |

### 7.2 量化推理

| 方法 | 位数 | 说明 |
|------|------|------|
| **AQLM** | 2-bit | 非均匀量化 |
| **AWQ** | 4-bit | 激活感知量化 |
| **GPTQ** | 4-bit | 后训练量化 |
| **LLM.int8()** | 8-bit | 混合精度量化 |
| **HQQ** | 2/3/4/8-bit | 半二次量化 |
| **EETQ** | 8-bit | 高效量化 |

### 7.3 推理后端

| 后端 | 参数 | 特点 |
|------|------|------|
| **HuggingFace** | `infer_backend=hf` (默认) | 标准推理 |
| **vLLM** | `infer_backend=vllm` | 高吞吐量推理 |
| **SGLang** | `infer_backend=sglang` | 高性能推理 (2025新增) |

### 7.4 分布式训练

- **DeepSpeed**: ZeRO-0/1/2/3, ZeRO-Infinity
- **FSDP**: PyTorch 原生全分片数据并行
- **Accelerate**: HuggingFace 分布式抽象层
- **Ray**: 弹性分布式训练
- **Megatron-core**: NVIDIA 高性能训练后端 (2025新增)
- **Ascend NPU**: 华为昇腾支持

---

## 8. 配置参数全解

### 8.1 ModelArguments (模型参数)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_name_or_path` | None | 模型路径或 Hub ID (必填) |
| `adapter_name_or_path` | None | 适配器路径，逗号分隔支持多个 |
| `trust_remote_code` | False | 信任远程代码 |
| `flash_attn` | auto | FlashAttention 模式 |
| `rope_scaling` | None | RoPE 缩放策略 |
| `use_unsloth` | False | 启用 Unsloth 加速 |
| `enable_liger_kernel` | False | 启用 Liger Kernel |
| `mixture_of_depths` | None | MoD 转换/加载 |
| `infer_backend` | hf | 推理后端 |
| `use_kv_cache` | True | KV Cache 加速生成 |

### 8.2 DataArguments (数据参数)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `template` | None | 对话模板名称 |
| `dataset` | None | 训练数据集名，逗号分隔 |
| `eval_dataset` | None | 评估数据集名 |
| `dataset_dir` | data | 数据集目录 |
| `cutoff_len` | 2048 | 最大 token 长度 |
| `packing` | None | 序列打包 |
| `streaming` | False | 流式数据加载 |
| `val_size` | 0.0 | 验证集比例 |
| `max_samples` | None | 最大样本数 (调试用) |
| `mask_history` | False | 只在最后一轮计算 loss |

### 8.3 FinetuningArguments (微调参数)

#### 通用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `stage` | — | 训练阶段 (pt/sft/rm/ppo/dpo/kto/orpo) |
| `finetuning_type` | — | 微调类型 (lora/oft/full/freeze) |

#### LoRA 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lora_rank` | 8 | LoRA 秩 |
| `lora_alpha` | None | 缩放因子 (默认 2×rank) |
| `lora_dropout` | 0.0 | LoRA Dropout |
| `lora_target` | all | 目标模块 |
| `use_rslora` | False | 秩稳定缩放 |
| `use_dora` | False | 权重分解 |
| `pissa_init` | False | PiSSA 初始化 |
| `loraplus_lr_ratio` | None | LoRA+ 学习率比率 |

#### RLHF 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `pref_beta` | 0.1 | 偏好损失 beta |
| `pref_loss` | sigmoid | DPO 损失类型 |
| `ref_model` | None | 参考模型路径 |
| `reward_model` | None | 奖励模型路径 (PPO) |
| `reward_model_type` | lora | 奖励模型类型 |
| `simpo_gamma` | 0.5 | SimPO 目标奖励边界 |

#### 优化器参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_galore` | False | GaLore 梯度优化 |
| `galore_rank` | 16 | GaLore 秩 |
| `use_apollo` | False | APOLLO 优化 |
| `use_badam` | False | BAdam 优化 |

### 8.4 TrainingArguments (训练参数)

继承自 `Seq2SeqTrainingArguments` (Transformers)，并扩展了：

| 参数 | 说明 |
|------|------|
| `ray_num_workers` | Ray 训练工作节点数 |
| `fp8` | 启用 FP8 训练 |
| `fp8_backend` | FP8 后端 (auto/torchao/te/msamp) |

---

## 9. 使用指南

### 9.1 安装

```bash
# 从 PyPI 安装
pip install llamafactory

# 从源码安装
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"

# Docker 安装
docker pull hiyouga/llamafactory:latest
```

### 9.2 CLI 快速开始

```bash
# SFT 微调
llamafactory-cli train \
    --model_name_or_path meta-llama/Llama-3-8B-Instruct \
    --finetuning_type lora \
    --stage sft \
    --dataset your_dataset \
    --template llama3 \
    --output_dir ./output

# DPO 偏好对齐
llamafactory-cli train \
    --model_name_or_path meta-llama/Llama-3-8B-Instruct \
    --adapter_name_or_path ./sft_checkpoint \
    --finetuning_type lora \
    --stage dpo \
    --dataset preference_data \
    --template llama3

# 模型推理
llamafactory-cli chat \
    --model_name_or_path meta-llama/Llama-3-8B-Instruct \
    --adapter_name_or_path ./output \
    --template llama3

# 启动 API 服务
llamafactory-cli api \
    --model_name_or_path meta-llama/Llama-3-8B-Instruct \
    --template llama3
```

### 9.3 YAML 配置示例

```yaml
### model
model_name_or_path: meta-llama/Llama-3-8B-Instruct
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

### dataset
dataset: alpaca_zh
template: llama3
cutoff_len: 1024
max_samples: 50000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: ./sft_output
logging_steps: 10
save_steps: 500
plot_loss: true

### train
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true

### eval
val_size: 0.1
per_device_eval_batch_size: 4
eval_strategy: steps
eval_steps: 500
```

```bash
# 使用配置文件训练
llamafactory-cli train config.yaml
```

### 9.4 Web UI (LLaMA Board)

```bash
# 启动 Gradio Web UI
llamafactory-cli webui

# 或者
python src/webui.py
```

Web UI 提供:
- 可视化模型选择
- 数据集配置
- 训练参数调整
- 实时训练监控 (TensorBoard)
- 一键导出/部署

### 9.5 API 部署

```bash
# OpenAI 兼容 API
llamafactory-cli api \
    --model_name_or_path ./merged_model \
    --template llama3 \
    --infer_backend vllm  # 可选 vllm 加速
```

```python
# 调用示例 (与 OpenAI API 兼容)
import openai
client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
response = client.chat.completions.create(
    model="llama3",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

---

## 10. 技术亮点与设计哲学

### 10.1 核心亮点

1. **真正的零代码**: 通过 Web UI 可以完成从数据准备到模型部署的全流程
2. **模型适配即插即用**: 通过配置文件即可支持新模型，无需改代码
3. **训练方法全覆盖**: 从预训练到 RLHF，一站式支持
4. **Day-0 新模型支持**: 对主流新模型提供首发支持
5. **多硬件适配**: NVIDIA GPU / AMD GPU / 华为 NPU / Intel 等
6. **内存效率极致**: 量化 + 梯度检查点 + 流式加载 + 序列打包

### 10.2 设计哲学

```
"Make fine-tuning as easy as training a classifier."
```

- **配置优于代码**: 90% 的场景不需要写代码
- **组合优于继承**: 模型/数据/训练方法自由组合
- **性能不妥协**: 积极集成 FlashAttention、Unsloth、vLLM 等加速方案
- **社区驱动**: Day-0 支持社区需要的新模型

### 10.3 工程质量

- **代码规范**: 使用 Ruff 进行 lint 和格式化
- **类型标注**: 全面使用 Python 类型提示
- **测试覆盖**: CI 自动测试
- **文档完善**: 中英文 README、官方博客、ReadTheDocs

---

## 11. 与其他框架对比

| 特性 | LLaMA Factory | Axolotl | unsloth | swift |
|------|:---:|:---:|:---:|:---:|
| 模型数量 | 100+ | ~20 | ~10 | 50+ |
| 零代码 WebUI | ✅ | ❌ | ❌ | ✅ |
| 训练方法全覆盖 | ✅ | ✅ | 部分 | ✅ |
| Day-0 模型支持 | ✅ | ❌ | ❌ | 部分 |
| 多模态支持 | ✅ | 部分 | ❌ | ✅ |
| 多硬件支持 | ✅ | 有限 | ❌ | ✅ |
| API 服务 | ✅ | ❌ | ❌ | ✅ |
| 社区活跃度 | 极高 | 高 | 高 | 中 |
| 学术认可 | ACL 2024 | 无 | 无 | 无 |

---

## 12. 总结与建议

### 12.1 适用场景

- ✅ **快速原型**: 想快速微调一个模型做实验
- ✅ **生产部署**: 需要从微调到部署的一站式方案
- ✅ **多模型对比**: 需要在同一框架下对比不同模型
- ✅ **多模态任务**: 需要处理图像/视频/音频
- ✅ **RLHF 流程**: 需要完整的偏好对齐流程
- ✅ **资源受限**: 需要 QLoRA 量化训练

### 12.2 注意事项

- 大模型全参数微调仍需要充足的 GPU 资源
- Web UI 适合实验，生产环境建议使用 YAML 配置 + CLI
- 注意 transformers 版本兼容性 (需在 4.51.0-5.2.0 之间)
- 使用 vLLM/SGLang 推理后端需要额外安装

### 12.3 学习路径建议

```
入门: Web UI 体验 → CLI 基础训练 → YAML 配置优化
进阶: 自定义数据集 → 自定义模板 → 多模态微调
高级: 分布式训练 → RLHF 全流程 → 模型部署优化
```

### 12.4 参考资源

- **GitHub**: https://github.com/hiyouga/LLaMA-Factory
- **文档**: https://llamafactory.readthedocs.io/
- **博客**: https://blog.llamafactory.net/
- **Colab**: https://colab.research.google.com/drive/1eRTPn37ltBbYsISy9Aw2NuI2Aq5CQrD9
- **论文**: "LLaMA Board: A Zero-code Platform for Fine-tuning Large Language Models" (ACL 2024)

---

> 📝 本报告由技术分析自动生成，基于 LLaMA Factory main 分支源码分析。
> 最后更新: 2026-03-17
