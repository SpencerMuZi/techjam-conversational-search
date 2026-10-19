# DeepSeek 真实 API 测试记录

日期：2026-10-20（Asia/Singapore）。模型：`deepseek-flash`；目录：50,000 个商品。

已通过官方 API 鉴权，并完成两次、共 8 轮真实对话调用。最终回归的 4 轮全部通过，测试期间关闭离线回退，因此成功结果来自真实 LLM 工具循环。

| 场景 | 结果 | 耗时 | 推荐数 | 实际工具顺序 |
| --- | --- | ---: | ---: | --- |
| 真皮运动鞋／预算100 | PASS | 7.84s | 3 | update_search_state → search_products → search_products → search_products → get_product_details → finish |
| 改为黑色／保留预算 | PASS | 9.52s | 0 | update_search_state → search_products → search_products → get_product_details → get_product_details → search_products → finish |
| 切换纯棉T恤／预算30 | PASS | 7.33s | 1 | update_search_state → search_products → search_products → get_product_details → finish |
| 需求不明确时主动追问 | PASS | 0.73s | 0 | ask_user |

最终 4 轮累计输入 105,394 tokens，输出 2,694 tokens。两次测试合计输入 240,721、输出 5,641 tokens；费用以账户账单为准。

## 实测发现并修复的问题

首轮实测中，模型虽然在文字中承认部分商品价格未知，却仍把它们放入硬预算推荐数组。修复后增加结构化 `max_price`，最终推荐工具会拒绝价格未知、无效或超上限的商品，要求模型修正后再输出。新增 4 项预算回归测试。

最终测试验证了预算保留、预算修改、品类切换清空旧要求、有效且唯一的商品 ID、实际 token 统计，以及主动追问。第二轮出现两次“引用当前未观察到的商品 ID”的工具调用，均被校验拒绝；模型读取错误观察后继续检索并正常结束，没有离线回退。

## 回归结果

- 完整测试套件：37/37 通过（使用已安装运行时和实验依赖的 Python 环境）。
- 200 个公开会话的离线基线：Hit@10=0.995，MRR=0.791，MTTC=2.51，TechnicalScore=0.9046，与修改前一致。
- 4 轮真实测试是小规模连通性与行为验证，不代表完整 LLM 检索质量评估；材质、颜色及回答事实仍需模型核验，程序硬预算校验也依赖需求被正确抽取。

## 本地运行

密钥只保存在本地 `.env`，不提交到 Git；原始凭据文档已通过本地 exclude 忽略。当前配置为 `react`，模型 `deepseek-flash`，`SHOPPING_COPILOT_LLM_FALLBACK=false`，便于暴露真实 API 错误。

在项目根目录执行：

```bash
set -a
source .env
set +a
python3 scripts/demo_session.py --mode react --interactive --trace
```

详细请求、响应、工具顺序和检查结果见 [JSON 记录](deepseek_live_smoke.json)。记录不包含密钥。
