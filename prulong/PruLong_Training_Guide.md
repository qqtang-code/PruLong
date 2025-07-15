# PruLong 训练全流程指南

这是一份详细的PruLong训练指南，涵盖从环境安装到数据准备再到模型训练的完整流程。即使是初学者也能按照本指南成功启动PruLong训练。

## 目录
- [1. 环境准备](#1-环境准备)
- [2. 数据准备](#2-数据准备)
- [3. 训练配置](#3-训练配置)
- [4. 启动训练](#4-启动训练)
- [5. 训练后处理](#5-训练后处理)
- [6. 故障排除](#6-故障排除)

## 1. 环境准备

### 1.1 创建conda环境

```bash
# 创建新的conda环境
conda create -n prulong python=3.11
conda activate prulong
```

### 1.2 安装基础依赖

```bash
# 进入prulong目录
cd /path/to/PruLong/prulong

# 安装基础依赖
pip install -r requirements.txt
```

### 1.3 安装Flash Attention（重要）

Flash Attention需要特殊的安装方式：

```bash
# 方法1：直接安装（推荐）
pip install flash-attn==2.6.1 --no-build-isolation

# 方法2：如果上述方法失败，从源码编译
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
python setup.py install
```

### 1.4 验证安装

```bash
# 验证PyTorch CUDA可用
python -c "import torch; print(torch.cuda.is_available())"

# 验证Flash Attention安装
python -c "import flash_attn; print('Flash Attention installed successfully')"

# 验证transformers版本
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
```

## 2. 数据准备

### 2.1 下载ProLong预处理数据（推荐）

最简单的方法是直接使用ProLong提供的预处理数据：

```bash
# 创建数据目录
mkdir -p datasets

# 下载64K训练数据
git clone https://huggingface.co/datasets/princeton-nlp/prolong-data-64K datasets/long-context-65536

# 下载512K训练数据（可选，用于更长序列训练）
git clone https://huggingface.co/datasets/princeton-nlp/prolong-data-512K datasets/long-context-524288

# 下载SFT数据
git clone https://huggingface.co/datasets/princeton-nlp/prolong-ultrachat-64K datasets/prolong-ultrachat-64K

# 创建sample数据用于快速测试
mkdir -p datasets/prolong-sample
cp -r datasets/long-context-65536/* datasets/prolong-sample/
```

### 2.2 数据结构说明

下载完成后，数据结构应该如下：

```
datasets/
├── long-context-65536/              # 64K预训练数据
│   ├── thestackv1_concat_by_repo-65536/
│   ├── book-65536/
│   ├── fineweb-edu/
│   └── ...
├── long-context-524288/             # 512K预训练数据
│   ├── thestackv1_concat_by_repo-524288/
│   ├── book-524288/
│   └── ...
├── prolong-ultrachat-64K/           # SFT数据
└── prolong-sample/                  # 测试用小样本数据
```

### 2.3 自定义数据准备（高级用户）

如果需要使用自己的数据：

```bash
# 安装datatools（用于数据处理）
pip install git+https://github.com/CodeCreator/datatools.git

# 将原始文本数据分词并打包成64K长度
# 输入数据应为mosaicml-streaming格式，每个样本包含"text"字段
pack input_data_path output_data_path --pack_length 65536 --min_length 1024 -w 40

# 或者先分词再打包
tokenize raw_text_data tokenized_data -w 40 --tokenizer meta-llama/Meta-Llama-3-8B
pack tokenized_data packed_data --pack_length 65536 --min_length 1024 -w 40
```

## 3. 训练配置

### 3.1 环境变量设置

创建训练环境配置文件 `setup_training_env.sh`：

```bash
#!/bin/bash

# 设置基础环境变量
export WANDB_PROJECT="prulong"
export WANDB_MODE="offline"  # 离线模式，避免网络问题
export TOKENIZERS_PARALLELISM=true
export LOGIT_BLOCK_SIZE=2048
export OMP_NUM_THREADS=8

# 内存优化
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo "Training environment setup complete!"
```

### 3.2 修改训练脚本

根据你的硬件配置修改训练脚本。以下是三种训练模式的配置：

#### 3.2.1 只训练mask参数（推荐用于已经指令调优的模型）

编辑 `run_scripts/prulong_masksonly.sh`：

```bash
# 模型配置
model="meta-llama/Llama-3.1-8B-Instruct"  # 或其他Llama模型
dataset="datasets/prolong-sample"          # 使用sample数据进行测试

# 硬件配置（根据你的GPU数量调整）
NUM_GPUS=4        # 你的GPU数量
NUM_NODES=1       # 节点数量
BSZ=8             # 批次大小，根据GPU显存调整
SEQ=1             # 每设备序列数

# 训练超参数
STEPS=100         # 步数，测试时可设置较小值
LR=1e-5           # 学习率
WARMUP=0.1        # 预热比例

# PruLong特定参数
MAX_TOKS=32768    # 每设备最大token数，根据显存调整
END_HEAD_SPARSITY=0.5  # 目标稀疏度
CONTEXT_WINDOW_IF_TOGGLED=1024  # 局部窗口大小
```

#### 3.2.2 同时训练mask和权重参数

编辑 `run_scripts/prulong_masksandweights.sh`：

```bash
# 与masksonly基本相同，但需要：
FREEZE_WEIGHTS=false    # 不冻结权重
FREEZE_MASKS=false      # 不冻结mask
START_HEAD_SPARSITY=0.7 # 起始稀疏度
END_HEAD_SPARSITY=0.7   # 结束稀疏度
```

#### 3.2.3 SFT训练（在已有PruLong模型基础上）

编辑 `run_scripts/sft.sh`：

```bash
# 使用已训练的PruLong模型作为基础
model="path/to/your/prulong/checkpoint"
dataset="datasets/prolong-ultrachat-64K"

# SFT特定设置
FREEZE_WEIGHTS=false    # 训练权重
FREEZE_MASKS=true       # 冻结mask
MASK_LEARNING_RATE=0    # mask学习率为0
REG_LEARNING_RATE=0     # 正则化学习率为0
```

## 4. 启动训练

### 4.1 准备运行目录

```bash
# 创建必要目录
mkdir -p checkpoints
mkdir -p joblog

# 确保脚本可执行
chmod +x run_scripts/*.sh
chmod +x setup_training_env.sh
```

### 4.2 单机训练启动

```bash
# 加载环境变量
source setup_training_env.sh

# 启动只训练mask的训练（推荐新手开始）
bash run_scripts/prulong_masksonly.sh

# 或者使用其他模式
# bash run_scripts/prulong_masksandweights.sh
# bash run_scripts/sft.sh
```

### 4.3 多机训练启动（SLURM集群）

如果你在SLURM集群上：

```bash
# 提交作业到队列
sbatch run_scripts/prulong_masksonly.sh

# 查看作业状态
squeue -u $USER

# 查看作业输出
tail -f joblog/prulong_masksonly-<JOB_ID>.out
```

### 4.4 监控训练进度

```bash
# 查看训练日志
tail -f checkpoints/<run_name>/log.out

# 查看GPU使用情况
nvidia-smi

# 查看检查点
ls -la checkpoints/<run_name>/
```

## 5. 训练后处理

### 5.1 保存训练得到的mask

训练完成后，提取并保存mask：

```bash
# 保存mask到TSV文件
python save_prulong_masks.py \
    --checkpoint checkpoints/<your_run_name> \
    --sparsity 0.7 \
    --out_path checkpoints/<your_run_name>/masks_sp0.7.tsv

# 不指定稀疏度（保存原始mask值）
python save_prulong_masks.py \
    --checkpoint checkpoints/<your_run_name> \
    --out_path checkpoints/<your_run_name>/masks_raw.tsv
```

### 5.2 模型评估

在PruLong根目录下使用evaluation脚本：

```bash
# 切换到评估目录
cd ../eval

# 安装评估依赖
pip install -r requirements.txt

# 运行评估（需要根据具体需求修改配置）
python eval.py --model_path ../prulong/checkpoints/<your_run_name>
```

## 6. 故障排除

### 6.1 常见错误及解决方案

#### 内存不足 (OOM)
```bash
# 解决方案：减少批次大小和最大token数
export BSZ=4          # 减少批次大小
export MAX_TOKS=16384 # 减少最大token数
export SEQ=1          # 确保每设备序列数为1
```

#### Flash Attention安装失败
```bash
# 解决方案：使用特定CUDA版本
pip uninstall flash-attn
pip install flash-attn==2.6.1 --no-build-isolation --force-reinstall

# 或者指定CUDA版本
FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn==2.6.1
```

#### 数据加载错误
```bash
# 检查数据路径
ls -la datasets/prolong-sample/

# 检查数据格式
python -c "
from streaming import LocalDataset
dataset = LocalDataset('datasets/prolong-sample')
print(f'Dataset length: {len(dataset)}')
print(f'Sample: {dataset[0].keys()}')
"
```

#### 分布式训练问题
```bash
# 检查NCCL
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# 或者禁用分布式训练进行调试
export NUM_GPUS=1
export NUM_NODES=1
```

### 6.2 调试技巧

#### 快速测试配置
```bash
# 使用小数据集和少步数进行快速测试
export DATASET="datasets/prolong-sample"
export STEPS=10
export SAVE_STEPS=5
export BSZ=2
export MAX_TOKS=4096
```

#### 检查GPU状态
```bash
# 监控GPU使用
watch -n 1 nvidia-smi

# 检查CUDA版本匹配
python -c "
import torch
print(f'PyTorch CUDA version: {torch.version.cuda}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
"
```

#### 日志分析
```bash
# 查找错误信息
grep -i "error\|exception\|failed" checkpoints/<run_name>/log.out

# 查看内存使用
grep -i "memory\|oom" checkpoints/<run_name>/log.out

# 查看训练进度
grep -i "step\|loss" checkpoints/<run_name>/log.out | tail -20
```

## 7. 高级配置

### 7.1 自定义训练参数

创建自定义配置文件 `custom_config.sh`：

```bash
#!/bin/bash

# 模型和数据配置
export MODEL="meta-llama/Llama-3.1-8B-Instruct"
export DATASET="datasets/your-custom-dataset"

# 训练超参数
export BSZ=16
export LR=2e-5
export STEPS=5000
export WARMUP=0.05

# PruLong特定参数
export START_HEAD_SPARSITY=0.0
export END_HEAD_SPARSITY=0.8
export MASK_LEARNING_RATE=1.0
export CONTEXT_WINDOW_IF_TOGGLED=2048

# 硬件配置
export NUM_GPUS=8
export FSDP=1              # FSDP策略
export GC=1                # 梯度检查点
export SEQ_PARALLEL_SIZE=2 # 序列并行大小
```

### 7.2 从检查点恢复训练

```bash
# 自动从最新检查点恢复
bash run_scripts/prulong_masksonly.sh

# 从指定检查点恢复
export RESUME_FROM_CHECKPOINT="checkpoints/<run_name>/checkpoint-1000"
bash run_scripts/prulong_masksonly.sh
```

### 7.3 多阶段训练

```bash
# 第一阶段：64K训练
export DATASET="datasets/long-context-65536"
export CONTEXT_WINDOW_IF_TOGGLED=1024
bash run_scripts/prulong_masksonly.sh

# 第二阶段：基于第一阶段结果进行512K训练
export MODEL="checkpoints/stage1_run_name"
export DATASET="datasets/long-context-524288"
export CONTEXT_WINDOW_IF_TOGGLED=4096
bash run_scripts/prulong_masksandweights.sh

# 第三阶段：SFT
export MODEL="checkpoints/stage2_run_name"
export DATASET="datasets/prolong-ultrachat-64K"
bash run_scripts/sft.sh
```

## 8. 性能优化建议

### 8.1 硬件要求
- **最低配置**：4x RTX 3090 (24GB)
- **推荐配置**：8x A100 (80GB)
- **内存**：每GPU至少32GB系统内存
- **存储**：SSD，至少1TB可用空间

### 8.2 性能调优参数
```bash
# 内存优化
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=0

# 数据加载优化
export DATALOADER_NUM_WORKERS=4
export DATALOADER_PIN_MEMORY=true

# 计算优化
export TORCH_COMPILE=true  # PyTorch 2.0编译加速
export BF16=true           # 使用bfloat16
```

这份指南涵盖了PruLong训练的完整流程。建议初学者先使用小数据集进行测试，确认环境配置正确后再进行完整训练。如有问题，请参考故障排除部分或查看原始论文和代码仓库。
