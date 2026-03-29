# hw04

大模型文案、剪映声音克隆与开源语音识别实践（HW04）

## 目录结构

```
hw04/
├── README.md                # 项目总览
├── text_gen.md              # 任务一：大模型生成文稿
├── jianying.md              # 任务二：剪映声音克隆说明
├── asr_report.md            # 任务三：ASR方案调研报告
├── experimentlog.md         # 任务三：实验日志
├── audio_recognition.py     # 任务三：音频文件识别代码
├── realtime_recognition.py  # 任务三：实时麦克风识别代码
└── requirements.txt         # 任务三：依赖文件
```

## 任务说明

### 任务一：大模型生成文稿

- **文件**：`text_gen.md`
- **内容**：包含标题、大模型生成的150字文本、模型与Prompt说明

### 任务二：剪映声音克隆

- **文件**：`jianying.md`
- **内容**：剪映声音克隆步骤概要、导出文件格式说明
- **输出**：配音音频文件（需手动生成）

### 任务三：开源语音识别调研与实现

- **文件**：
  - `asr_report.md`：三种及以上ASR方案对比、选型理由
  - `experimentlog.md`：实验过程与结果记录
  - `audio_recognition.py`：音频文件识别代码
  - `realtime_recognition.py`：实时麦克风识别代码
  - `requirements.txt`：依赖文件

## 运行说明

### 任务三运行步骤

1. **环境准备**：
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **安装ffmpeg**（Whisper依赖）：
   - 下载ffmpeg并添加到系统环境变量
   - 验证安装：`ffmpeg -version`
3. **音频文件识别**：
   ```bash
   python audio_recognition.py <音频文件路径> --model <模型名称>
   ```
   示例：
   ```bash
   python audio_recognition.py voice_clone_output.mp3 --model base
   ```
4. **实时麦克风识别**：
   ```bash
   python realtime_recognition.py
   ```

## 注意事项

- 任务二需要在本地安装剪映软件并手动完成声音克隆
- 实时麦克风识别需要麦克风设备支持

