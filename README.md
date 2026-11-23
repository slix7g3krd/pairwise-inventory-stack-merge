# pairwise-inventory-stack-merge

## 领域

游戏、图形与多媒体

## 主题

背包堆叠合并

## 说明

并发合并、上限变更与离线补发。

初始状态下 `gameplay/inventory_stack_merge.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
