# 代码来源、适配与上游贡献

## 独立项目中的代码

| 材料 | 来源与用途 | 本次检查 |
| --- | --- | --- |
| 历史专家计数器 | 2026-11-04 模型实例绑定版，原始字节保存在 `reference/` | SHA 校验；合成数据的保存、恢复、身份错误、损坏和并发最终刷盘 |
| 离线分析工具 | 提取 STRC/STRP 格式和密度排序，显式输入账本、布局、预算 | 19 项 Python 测试覆盖工具及历史计数器的有限路径 |
| GU256 patch | 回放保存的本地转换，基于上游 `46c152a1` | 对固定源文件执行 patch application check；CUDA 编译与 GPU 数值检查沿用历史范围 |
| 提交前缀 helper | 保存的实际 C++ helper 与组件测试 | 本次 Mac CPU 执行 440 个预算/EOS/前缀案例 |
| BYIELD 统计 patch | 从历史修复移植至上游 `9f7d977d` | 当前上游模拟引擎回归与相邻 PR 兼容检查 |

离线工具有两项明确适配。密度比较使用 Fraction 保留大整数精度，层内索引使用取模以处理非 2 的幂几何。历史计数器保持原字节供阅读，原先以 512 专家为主要部署条件。

离线导出的是纯全局密度画像。历史三层 RAM 补集布局包含 guard 与层内重排，使用其加载规则时应核对完整历史实现。生成的画像通过明确步骤加载，引擎端兼容性需要按具体版本确认。

历史在线采集入口仍包含部署默认值、Windows ctypes 调用与进程全局 manager。它保留了当时行为，在线命名管道路径未在此次发布阶段重新执行。新发布的命令行工具只调用独立的只读分析模块。

Strata 提供 GPU 专家缓存、原生适应性晋升、KV 与调度机制。本项目的贡献包括累计统计、预算画像、模型身份隔离、针对硬件的实验与相应修复。模型及其他第三方资产分别遵循其许可证。

## Strata 官方 PR

[PR #5318 — Preserve external cache and sum prefill work across BYIELD](https://github.com/Niko1221/Strata/pull/5318)

作者为 `YusenZhang0601`，提交 `82ffb65`，基线 `9f7d977d220e9fdf2ec53cf2620cdadac1cfb24e`。PR 只修改 Python 服务统计并添加回归，现有让行次数保留。

### 复现的现象

Mac 上使用真实 Service、StrataEngine 和 Python 模拟 native producer，运行 4 个新回归。官方基线有 3 个断言失败、1 个通过。例如原本外部缓存只有 3 tokens，后续内部恢复却被报告为 384 tokens。取消两次让行时的 input/cache/read 为 `(128,64,64)`，预期 `(128,3,125)`。

修复保存已经让行的 DONE，汇总各段 prompt 工作，并在取消恰好落到同一个 DONE 对象时避免重复累计。

### 当前移植的验证

环境为 macOS arm64、Python 3.14.6、独立虚拟环境与仓库固定的 regex/Jinja2。

```sh
python -B -m unittest serve.test_yield_accounting serve.test_request_accounting -v
python -B -m unittest serve.test_parallel serve.test_yield_accounting serve.test_request_accounting -v
```

- 第一组 14 项通过，10.783 秒。
- 第二组 62 项通过，59.380 秒。
- 在相邻 [PR #5307](https://github.com/Niko1221/Strata/pull/5307) 的 `08519d46` 上重新应用修复，72 项通过，65.026 秒。

两项 PR 在 finalizer 有文本重叠。首次直接应用补丁失败，连带缺失测试模块；重新应用后完成上述兼容检查。初次本地测试还遇到依赖缺失，随后在项目独立环境安装固定依赖并运行。各次尝试在发布验证收据中分别记录。

此次上游移植尚未执行 GPU 模型、Windows runtime 或原生后端构建。测试围绕服务层统计正确性。

## 本地补丁的复用

`patches/strata-prefill-yield-accounting.patch` 包含服务代码改动，`patches/test_yield_accounting.py` 为同提交的新回归。官方 PR 持有完整差异、审阅与 CI 状态。

`patches/gongwei-gu256.json` 记录固定 CUDA 源文件、转换前后 SHA 与 opt-in 开关。该 patch 是历史实验的公开材料，应用到更新源码前先核对版本并重做数值 oracle。

`patches/accepted_commit.hpp` 仅包含提交边界 helper，完整集成还需对应串行/流水线调用点及源码基线。其组件测试命令如下。

```sh
c++ -std=c++17 -Wall -Wextra -Werror patches/accepted_commit_test.cpp -o /tmp/accepted-commit
/tmp/accepted-commit
```

Mac 本次需在编译命令中显式传入本机已安装 SDK 的 sysroot 和 C++ 头文件目录。首次默认 clang++ 命令因未找到标准头文件而失败；显式路径下 440 项通过。CUDA kernel 的运行时正确性和性能仍按历史 GPU 验证范围表述。
