# 更新日志

## v1.1.1

### 修复

- `sentence` 分割模式扩展匹配范围：在句号基础上，新增问号（`？?`）和感叹号（`！!`）作为句子边界。

## v1.1.0

### 新增

- 新增配置 `judge.emotion_scope`，支持三种情绪计算范围：
  - `whole`：整段话（原有行为）
  - `punctuation`：按标点符号切割，每段独立计算情绪
  - `sentence`：按句号/问号/感叹号切割成句，可配合 `judge.sentence_group_size` 按组计算情绪
- 新增配置 `judge.sentence_group_size`，控制 `sentence` 模式下每组几句

### 修复

- 修复分段推理时情绪缓存跨段复用导致所有段情绪相同的问题
- 修复 `说` / `gsv` 命令不走情绪逻辑的问题
- 修复音频缓存目录不存在时保存失败（`No such file or directory`）的问题

## v1.0.0

### 新增

- 基于原作者 Zhalslar 的 astrbot_plugin_GPT_SoVITS 插件进行修改和优化
- 更新插件信息和作者信息
- 添加致谢信息

## v3.1.0

### 新增

- 新增 `LocalDataManager`，统一管理本地音频读写与缓存逻辑。
- 新增缓存配置项：
  - `cache.enabled`：是否启用参数级缓存。
  - `cache.expire_hours`：缓存过期时间（小时，`0` 表示永不过期）。
  - `cache.path`：缓存目录。

### 优化

- TTS 推理流程支持“先查缓存再请求”：
  - 参数一致时直接复用本地音频，跳过重复推理请求。
  - 未命中时请求 GPT-SoVITS，成功后自动落盘供后续复用。
- 音频文件命名改为参数哈希命名：`gsv_<hash>.<ext>`，确保同参数稳定命中同一文件。
- 发送流程优先使用本地文件路径发送语音，无法读取时回退为 Base64 发送。
- `GSVRequestResult` 增加文本与缓存文件路径信息，便于链路透传和调试。
