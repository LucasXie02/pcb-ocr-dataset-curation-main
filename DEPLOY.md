# 部署指南

## 环境要求

- Python 3.10+
- 约 500MB 磁盘空间（不含数据和模型）

## 快速部署（仅审核工具）

如果只需要运行审核 Web 界面（不需要跑检测 pipeline），安装很轻量：

```bash
# 1. 克隆代码
git clone https://github.com/DaoAI-Robotics-Inc/pcb-ocr-dataset-curation.git
cd pcb-ocr-dataset-curation

# 2. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 启动
python ocr_review_app.py --port 5001
```

启动后浏览器打开 `http://localhost:5001`，通过界面上的「选择数据集」按钮指定数据目录。

或者直接指定数据目录启动：

```bash
python ocr_review_app.py --dataset_root /path/to/dataset --port 5001
```

## 完整部署（含检测 pipeline）

如果需要从 ZIP 文件生成标注数据，需要额外安装 PyTorch 和检测模型：

```bash
# 1. 安装审核工具依赖
pip install -r requirements.txt

# 2. 安装 pipeline 依赖
pip install torch torchvision rfdetr supervision tqdm pyyaml

# 3. 准备模型 checkpoint
#    放到 Checkpoints/ 目录下：
#    Checkpoints/char_224/config.json + model.pth
#    Checkpoints/char_448/config.json + model.pth
#    Checkpoints/orientation_classification/config.json + model.pth

# 4. 运行 pipeline
python process_all_zips.py \
    --zip-dir /path/to/zips/ \
    --work-dir /path/to/work/ \
    --output /path/to/output/
```

## 数据目录结构

审核工具期望的数据目录结构：

```
dataset_root/
  BoardA_fused/
    candidate/              ← 待审核
      ComponentClass_ID.json
      ComponentClass_ID.png
    final/                  ← 已通过（审核后自动创建）
    crop_manifest.json      ← 可选，有拼板时存在
  BoardB_fused/
    candidate/
      ...
  review.db                 ← 自动创建，审核事件记录
```

每个 `_fused` 目录是一块板。`candidate/` 里的 JSON + PNG 成对出现，JSON 是 LabelMe 格式的标注文件。

## 常用启动参数

```bash
python ocr_review_app.py \
    --dataset_root PATH    # 数据集根目录（可选，可在界面中选择）
    --class_name NAME      # 初始加载的板子名（默认第一个）
    --subdir candidate     # 初始子目录：candidate 或 final
    --port 5001            # 端口号
    --host 0.0.0.0         # 监听地址（默认所有网卡）
    --debug                # 调试模式（自动重载模板）
```

## 多人协作

Flask 开发服务器是单线程的，同一时间只能一个人操作。如果需要多人同时审核不同板子，可以启动多个实例（不同端口）指向同一个数据目录：

```bash
python ocr_review_app.py --dataset_root /data/ocr --port 5001  # 用户A
python ocr_review_app.py --dataset_root /data/ocr --port 5002  # 用户B
```

注意：不同用户应审核不同的板子，避免同时编辑同一张图。

## 故障排查

| 问题 | 解决 |
|------|------|
| `Address already in use` | `lsof -ti:5001 \| xargs kill -9` 然后重启 |
| 缩略图不显示 | 检查 PNG 文件是否与 JSON 在同一目录 |
| 组件总览为空 | 确认 `candidate/` 目录有 JSON+PNG 文件 |
| 编辑保存报错 | 检查文件写权限，确认 JSON 格式正确 |
| 板子显示黄点但已全部审核 | 点「完成审核 → final/」将文件移到 final/ |
