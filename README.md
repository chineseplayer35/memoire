# Memoire

**用结构代替算力的轻量长期记忆。**

不为每条消息调用模型，不做向量库，不做 RAG。给 bot 一个**像人一样的记忆结构**：当下上下文、每日事记、印象卡、日记。每天几分钱，换来"记得住人"的能力。

## 它是什么

一个 AstrBot 插件参考实现 + 一套记忆架构。核心思想只有一句：**记忆的关键是信息怎么组织，而不是算力怎么堆。**

- **当下上下文**：最近 50 条消息，4 小时无对话先提炼再裁剪
- **会话事记**：每个会话每天一条，最近 7 条常驻，更早的按日期召回
- **印象卡**：每个有过互动的人一份（≤240 字，初印象→改观，随聊天逐点演化），人在场自动带、被提及工具取
- **日记**：重要的事（约定/承诺/情绪节点），最近 10 条常驻，主动记录 + 每日定时梳理

## 为什么不用向量库 / 高频 LLM 调用

- 记忆的热路径（每次回消息）**零模型调用**：注入由确定性规则决定（人在场给卡、最近 7 条常驻），不靠语义检索，不会答非所问。
- 后台记忆更新只在**每天 22:30 和 4 小时空闲前**触发，且**没有新消息就一分钱不花**；模型也可按需主动更新印象卡或日记。
- 提炼走**游标 + 指纹增量处理**，只处理"上次提炼以来的新消息"，不重复烧钱。
- 本地文件存储（Markdown + JSON），可查看、可编辑、可备份，没有数据库依赖。

## 环境要求

- Python 3.10+
- 与插件所用 API 兼容的 AstrBot
- 当前仅声明支持 `aiocqhttp`，其他平台请自行验证

## 快速开始

1. 把 `plugin/` 中的全部文件复制到 AstrBot 的 `data/plugins/bot2_memory/` 目录。
2. 把 `config.example.json` 复制为 `data/config/bot2_memory.json`，按需修改；`summary_provider` 留空时使用当前会话的模型服务。
3. 重启 AstrBot，插件会注册五个工具：`memory_read_impression`、`memory_read_entries`、`memory_update_impression`、`memory_write_diary`、`memory_read_diary`。
4. 在人格提示词里写清工具使用规则（见 `docs/persona_example.md`）。

## 验证

在 AstrBot 根目录、已安装 AstrBot 依赖的 Python 环境中运行：

```bash
python -X utf8 data/plugins/bot2_memory/smoke_test.py
```

测试使用临时目录和模拟模型服务，不访问网络，也不读写正式记忆数据。当前版本已在 Python 3.12 + AstrBot 4.29.3 下通过冒烟测试。

## 隐私与数据

运行后，QQ 号、昵称及聊天提炼内容会保存在 AstrBot 的 `data/memory/` 中。请妥善控制这些文件的访问权限，不要提交到公开仓库；分享备份或日志前也应先去除个人信息。

## 目录

- `plugin/`：插件参考实现（`main.py`、`metadata.yaml`、`smoke_test.py`、`config.example.json`）
- `docs/`：架构、成本、人格示例
- `examples/`：印象卡、事记、日记的样例文件

## 许可证

MIT。随便用，改改也行，记得署个名。
