---
title: Qwen-0.5B-Chat LoRA 微调实践
date: 2025-07-13
categories: [人工智能, 大语言模型]
tags: [Qwen, LoRA, 微调, 大模型, PEFT]
math: true
---

## 一、项目背景与概述

### 1.1 项目目标
在本地服务器（NVIDIA RTX 4080 Super 16GB）上实现 Qwen-0.5B-Chat 大模型的 LoRA 微调，掌握大模型微调的项目结构、数据格式、依赖管理和训练流程。

### 1.2 技术环境
- **开发方式**：VSCode SSH 远程开发
- **操作系统**：Linux 环境，Ubuntu 22.04
- **虚拟环境**：Miniconda 虚拟环境
- **核心依赖**：PyTorch + Transformers + bitsandbytes + peft
- **硬件配置**：
  - CPU：i9-13900K
  - GPU：RTX 4080 Super 16GB
  - 内存：128GB
  - 存储：3.7TB NVMe SSD
- **软件版本**：
  - NVIDIA驱动：535
  - Python：3.10
  - PyTorch：2.3.0+cu121

## 二、项目结构与数据准备

### 2.1 项目目录结构
```plaintext
llm_finetune_demo/
├── finetune_qwen.py            # 微调主脚本
├── README.md
├── requirements.txt
├── data/
│   └── train.jsonl             # 训练数据
├── results/                    # 训练输出目录
```

### 2.2 数据格式规范
训练数据位于 `data/train.jsonl`，每行为一个 JSON 对象，包含以下字段：

```json
{
  "instruction": "写一首关于春天的诗。",
  "input": "",
  "output": "春风吹拂百花开，柳绿桃红燕归来。万物复苏生机现，江南水暖鸭先知。"
}
{
  "instruction": "将下面这句话翻译成英文。",
  "input": "我喜欢学习人工智能。",
  "output": "I like studying artificial intelligence."
}
```

## 三、Tokenizers 技术详解

### 3.1 核心架构组件

| 组件 | 提供方 | 作用 | 重要特性 |
|------|--------|------|----------|
| transformers | HuggingFace | 模型加载与训练框架 | 统一接口支持多种架构 |
| Model Hub | HuggingFace | 模型托管平台 | 社区贡献，版本管理 |
| tokenizers | HuggingFace | 基础分词库 | 支持多语言分词 |

### 3.2 分词与嵌入技术解析

#### 3.2.1 分词（Tokenization）
- **定义**：将文本拆分为模型可处理的最小单元（Token）
- **方法**：字级分词、子词分词（BPE、WordPiece、SentencePiece）
- **流程**：输入文本 → 分词器拆分为Token → Token映射为整数ID

#### 3.2.2 嵌入（Embedding）
- **定义**：将Token ID映射为连续向量的过程
- **数学表示**：

  $$
  \mathbf{E} \in \mathbb{R}^{|V| \times d}
  $$

  其中 $V$ 为词表大小， $d$ 为嵌入维度。

- **流程**：

  1. Token ID序列 $[t_1, t_2, ..., t_n]$
  2. 查表获取嵌入向量 $[\mathbf{e}_1, \mathbf{e}_2, ..., \mathbf{e}_n]$

### 3.3 分词与嵌入工作流

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 初始化模型和分词器
model_name = "Qwen/Qwen2-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16)

# 确保pad_token设置正确
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 编码阶段：文本 → Token → 向量
input_text = "猴子喜欢吃香蕉，兔子喜欢吃"
inputs = tokenizer(input_text, return_tensors="pt")
input_ids = inputs["input_ids"]

print("\n=== 分词结果 ===")
print(f"原始文本: '{input_text}'")
print(f"Token IDs: {input_ids.tolist()[0]}")
print(f"Token映射: {[tokenizer.decode(tok) for tok in input_ids[0]]}")

# 获取嵌入向量
with torch.no_grad():
    embeddings = model.get_input_embeddings()(input_ids)
print("\n=== 向量化结果 ===")
print(f"嵌入向量形状: {embeddings.shape}")
print(f"第一个Token的向量（前5个值）: {embeddings[0,0,:5].tolist()}...")

# 模型推理
with torch.no_grad():
    outputs = model(input_ids)
logits = outputs.logits

print("\n=== 模型输出 ===")
print(f"Logits形状: {logits.shape}")
print(f"最后一个位置的Top-5预测Token:")
last_token_logits = logits[0, 1, :]
topk = torch.topk(last_token_logits, 5)
for i, (prob, idx) in enumerate(zip(topk.values.softmax(-1), topk.indices)):
    print(f"{i+1}. {tokenizer.decode(idx)} (概率: {prob:.3f})")

# 解码阶段
predicted_id = torch.argmax(last_token_logits).item()
generated_text = tokenizer.decode([predicted_id])
print("\n=== 解码结果 ===")
print(f"预测的下一个Token: '{generated_text}'")
```

### 3.4 嵌入层特性说明

#### 3.4.1 是否通用/统一？
- **不是通用/统一的**
- 每个模型有自己专属的嵌入层和词表
- 不同模型的嵌入参数、维度、词表编号都不同

#### 3.4.2 嵌入与模型关系
- 嵌入层是模型结构的一部分
- 嵌入权重随模型训练而优化，与Transformer网络紧密耦合
- 不同模型架构即使嵌入维度相同，也不能直接通用

#### 3.4.3 嵌入与分词器关系
- 嵌入层大小和分词器词表大小严格对应
- 分词器决定Token到ID的映射，嵌入层根据ID查表获得向量


## 四、LoRA 原理与应用

### 4.1 技术概述
LoRA（Low-Rank Adaptation）是一种针对大模型的参数高效微调（PEFT, Parameter-Efficient Fine-Tuning）方法，通过引入少量可训练参数实现模型对新任务的快速适应。

### 4.2 核心原理
LoRA 在模型的部分权重矩阵上插入低秩可训练矩阵，将权重更新表示为低秩分解：

$$
W = W_0 + \Delta W = W_0 + BA
$$

其中：
- $W_0$：原始权重（冻结不变）
- $A, B$：新增的低秩可训练矩阵，秩远小于原始权重维度

### 4.3 方案对比

| 特性 | 全参数微调 | LoRA微调（PEFT） |
|------|------------|------------------|
| 依赖库 | transformers, datasets | transformers, datasets, peft |
| 参与训练参数 | 全部参数 | 只训练LoRA adapter参数 |
| 显存需求 | 高 | 低 |
| 训练速度 | 慢 | 快 |
| 保存模型大小 | 大（几百MB到几十GB） | 小（几MB到几百MB） |
| 适用场景 | 资源充足、数据量大、需大幅调整 | 资源有限、数据少、需多任务快速适配 |

### 4.4 技术优势
- **训练速度快**：仅更新小规模参数
- **易于部署**：推理时只需加载基础模型和 LoRA 增量参数
- **适合大模型**：尤其适合超大参数量的模型微调



## 五、训练方案对比与实现

### 5.1 全参数微调代码实现

```python
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# 1. 环境配置
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True

# 2. 模型与分词器加载
model_name = "Qwen/Qwen2-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
config.pretraining_tp = 1
config.use_cache = False

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16
)
model.resize_token_embeddings(len(tokenizer))

# 3. 数据预处理
def format_prompt(instruction, input=None, output=None):
    prompt = "<|im_start|>system\n你是一个AI助手<|im_end|>\n"
    prompt += f"<|im_start|>user\n{instruction}"
    if input:
        prompt += f"\n{input}"
    prompt += "<|im_end|>\n"
    if output:
        prompt += f"<|im_start|>assistant\n{output}<|im_end|>"
    return prompt

dataset = load_dataset("json", data_files="data/train.jsonl", split="train")
max_length = min(len(tokenizer(format_prompt(**dataset[0]))["input_ids"]), 1024)

def tokenize_function(batch):
    texts = [format_prompt(inst, inp, out) for inst, inp, out in zip(
        batch["instruction"], 
        batch.get("input", [""]*len(batch["instruction"])), 
        batch["output"]
    )]
    return tokenizer(texts, truncation=True, max_length=max_length, padding="max_length")

tokenized_data = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

# 4. 训练参数配置
training_args = TrainingArguments(
    output_dir="./qwen2-0.5b-finetuned",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=10,
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    bf16=True,
    fp16=False,
    logging_steps=10,
    save_strategy="epoch",
    optim="adamw_torch",
    gradient_checkpointing=True,
    report_to="none",
    remove_unused_columns=False,
    max_grad_norm=0.1,
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

# 5. 训练执行
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

print(f"训练前显存占用: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
trainer.train()
model.save_pretrained("./qwen2-0.5b-finetuned")
tokenizer.save_pretrained("./qwen2-0.5b-finetuned")
print(f"训练完成！峰值显存: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
```

### 5.2 LoRA 微调代码实现

```python
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 1. 环境配置
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True

# 2. 模型与分词器加载
model_name = "Qwen/Qwen2-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
config.pretraining_tp = 1
config.use_cache = False

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16
)
model.resize_token_embeddings(len(tokenizer))

# 3. LoRA配置与应用
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

# 准备模型
model = prepare_model_for_kbit_training(model)

# LoRA配置
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 应用LoRA
model = get_peft_model(model, lora_config)
print_trainable_parameters(model)

# 4. 数据预处理（同全参数微调）
def format_prompt(instruction, input=None, output=None):
    prompt = "<|im_start|>system\n你是一个AI助手<|im_end|>\n"
    prompt += f"<|im_start|>user\n{instruction}"
    if input:
        prompt += f"\n{input}"
    prompt += "<|im_end|>\n"
    if output:
        prompt += f"<|im_start|>assistant\n{output}<|im_end|>"
    return prompt

dataset = load_dataset("json", data_files="data/train.jsonl", split="train")
max_length = min(len(tokenizer(format_prompt(**dataset[0]))["input_ids"]), 1024)

def tokenize_function(batch):
    texts = [format_prompt(inst, inp, out) for inst, inp, out in zip(
        batch["instruction"], 
        batch.get("input", [""]*len(batch["instruction"])), 
        batch["output"]
    )]
    return tokenizer(texts, truncation=True, max_length=max_length, padding="max_length")

tokenized_data = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

# 5. 训练参数配置
training_args = TrainingArguments(
    output_dir="./qwen2-0.5b-finetuned",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=10,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    bf16=True,
    fp16=False,
    logging_steps=10,
    save_strategy="epoch",
    optim="adamw_torch",
    gradient_checkpointing=True,
    report_to="none",
    remove_unused_columns=False,
    max_grad_norm=0.1,
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

# 6. 训练执行
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

print(f"训练前显存占用: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
trainer.train()
model.save_pretrained("./qwen2-0.5b-finetuned")
tokenizer.save_pretrained("./qwen2-0.5b-finetuned")
print(f"训练完成！峰值显存: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
```

## 六、环境安装与依赖管理

### 6.1 关键环境配置要点

- **驱动与CUDA**：确保显卡驱动、CUDA、PyTorch版本匹配
- **bitsandbytes**：使用 `pip install bitsandbytes` 安装官方最新版
- **PyTorch**：使用官方CUDA支持版本
- **依赖包**：按需安装缺失包（如 `einops`, `transformers_stream_generator` 等）

### 6.2 requirements.txt 参考配置

```plaintext
# ==================== Python版本 ====================
# 建议使用 Python 3.10.18
# 建议用 conda 或 pyenv 管理

# ==================== CUDA/驱动（pip不负责） ====================
# 需提前安装NVIDIA驱动 >= 535.230.02
# CUDA驱动版本 >= 12.2  (nvidia-smi 检查)
# CUDA Toolkit 建议 12.1 或 12.2（pip不装CUDA toolkit，需系统自备）

# ==================== PyTorch（需和CUDA版本匹配） ====================
torch==2.3.0+cu121   # pip官方源装不到，需用PyTorch官网wheel或conda

# ==================== FlashAttention（需和CUDA匹配） ====================
flash-attn==2.3.6    # pip可装，需CUDA 12.1/Ampere+显卡

# ==================== HuggingFace生态 ====================
transformers==4.41.2
accelerate==1.8.1
peft==0.6.0
datasets==3.6.0
huggingface_hub==0.33.0
tokenizers==0.19.1
safetensors==0.5.3

# ==================== 其他依赖 ====================
numpy==2.1.2
tqdm==4.67.1
packaging==24.2
psutil==5.9.0
```

### 6.3 主要问题及解决方法

1. **分词器 pad_token 问题**
   - 解决方案：`tokenizer = AutoTokenizer.from_pretrained(..., pad_token='<|endoftext|>')`
   - 注意：不要使用 `add_special_tokens`，Qwen 分词器不支持！

2. **bitsandbytes 无GPU支持**
   - 检查 PyTorch 和 bitsandbytes CUDA 版本一致性
   - 使用最新版 bitsandbytes
   - 确保 `torch.cuda.is_available()` 返回 True

3. **模型权重下载缓慢**
   - 首次运行需耐心等待，后续使用本地缓存

4. **显存不足处理**
   - 调小 `per_device_train_batch_size`
   - 缩短 `max_length` 参数





