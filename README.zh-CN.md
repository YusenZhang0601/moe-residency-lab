# MoE Residency Lab

这个项目整理我在异构硬件上部署、剖析与第一性原理优化大型稀疏 MoE 模型的实测研究，公开硬件物理上限账本、专家统计与离线验证代码、源码优化数据与诚实证否记录。

- **`v0.2`（2026-11）**：在 RTX A6000 48 GB 上完成五场景物理硬件极限解剖与 6 大默认关闭源码优化（另诚实证否并剔除 `RFC-S1-L1`），合流发布 [`YusenZhang0601/Strata@personal-0.1.56-p2`](https://github.com/YusenZhang0601/Strata/tree/personal-0.1.56-p2)（`a1c2c904`）：4×100K 纯解码单窗 **`41.92 → 34.32 ms`（`-18.1%`）**、4×100K 端到端突发 **`20.55 → 11.92 s`（`1.72×`）**、100K 热增量 `155 tok` **`-22.8%`**（零借槽）、100K 冷停放 **`5.7×`**（`resize 701→0.4 ms`）、167K 宿主层 **`-53.4%`**，全开稳态剩余显存 `4,703–4,723 MiB`、`10/10` 验收全绿。
- **`v0.1`（2026-10-30 ~ 11-04）**：记录两台机器的早期部署取舍与纯标准库离线专家画像工具。

## 阅读顺序

| 想了解的内容 | 文档 |
| --- | --- |
| **v0.2：A6000 第一性原理硬件解剖、6 大优化、全栈大考与证否清单** | **[A6000 v0.2 深研报告（中文）](docs/a6000-first-principles-v2.zh-CN.md)** / **[English Report](docs/a6000-first-principles-v2.md)** |
| v0.1：16 GiB 显存、64 GiB RAM 的单路取舍 | [工位案例](docs/gongwei.md) |
| v0.1：A6000 的专家预算、缓存与四路体验 | [A6000 案例](docs/a6000.md) |
| 累计路由、字节热度与预算画像 | [专家统计与分配](docs/expert-residency.md) |
| 如何解读速度、缓存和失败 | [测量方法](docs/methodology.md) |
| 代码来源、补丁、`personal-0.1.56-p2` 与官方投稿 | [代码与上游贡献](docs/code-and-upstream.md) |

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
