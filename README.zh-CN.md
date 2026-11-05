# MoE Residency Lab

这个项目整理两台机器上部署、排查和优化大型 MoE 模型的实际经验，公开现有专家统计与分配代码、关键改动、测量结果和失败记录。

一期围绕三个问题展开。

1. 显存、内存和 SSD 容量有限时，专家和会话缓存怎样分配。
2. 长上下文与多个 Agent 同时使用时，时间究竟花在预填充、解码、恢复还是排队。
3. 哪些配置或代码改动有实测依据，哪些尝试没有带来可采用的收益。

## 阅读顺序

| 想了解的内容 | 文档 |
| --- | --- |
| 16 GiB 显存、64 GiB RAM 的单路取舍 | [工位案例](docs/gongwei.md) |
| A6000 的专家预算、缓存与四路体验 | [A6000 案例](docs/a6000.md) |
| 累计路由、字节热度与预算画像 | [专家统计与分配](docs/expert-residency.md) |
| 如何解读速度、缓存和失败 | [测量方法](docs/methodology.md) |
| 代码来源、补丁与官方投稿 | [代码与上游贡献](docs/code-and-upstream.md) |

## 在普通电脑上运行

在仓库根目录运行以下命令。需要 Python 3.10 或更新版本，核心工具使用标准库。

```sh
python examples/make_fixture.py demo-output
python -m moe_residency --ledger demo-output/counts.bin --layout demo-output/layout.json --budgets-gib 0.008 0.016 0.032 --export-profile demo-output/profile.bin
python examples/verify_results.py
python -m unittest discover -s tests -v
```

示例会创建独立的合成账本与画像。分析程序读取账本、比较排序策略，仅在显式指定导出路径时创建画像；目标已存在会报错。运行结束后可以删除 `demo-output`。

## 一期的证据边界

公开报告覆盖 2026-10-30 至 2026-11-04 的历史实验。每个阶段的模型、构建与配置分别标注。生产服务随后可能继续演变，复用时应重新核对版本。

已有实测支持专家预算、KV 布局与调度存在多目标取舍。学习画像相对默认画像的独立收益，还需要保持其他条件一致、使用后续负载的对照。仓库保留这一缺口，并提供离线比较所需的代码。

历史计数器完整源码单独保存在 `reference/`，在线 Windows 管道路径沿用当时实现。此次发布在 Mac 上验证离线工具与模拟引擎回归，正在运行的两台模型服务均保持原状。
