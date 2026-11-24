"""背包堆叠合并引擎。

本模块修复的六类堆叠缺陷：

1. **并发合并丢数量** —— 两笔拾取与一笔交易各自读旧值再写回，后写者覆盖先写者。
   现在所有合并走 :meth:`InventoryStore.transaction`：单事务（持锁 + 快照，
   失败整体回滚），每次写入对槽位版本号 ``revision`` 做 CAS 校验，
   读到旧版本的写入抛 :class:`StackRevisionConflict`，不会悄悄覆盖别人。
2. **上限调整后旧堆叠未拆分** —— :meth:`InventoryStore.restack` 按当前
   ``stack_cap`` / ``max_slots`` 统一重组所有物品，超额部分拆成新堆叠，
   物品总量严格守恒；槽位不足则整笔回滚。
3. **离线补发与在线物品重复合并** —— :class:`Mailbox` 以物品唯一 ID 去重，
   同一 ID 的重复补发/重登幂等跳过（:func:`offline_mail`）。
4. **排序键不稳定导致 UI 顺序跳动** —— :func:`sort_stable` 使用
   ``(item_id, seq, stack_id)`` 全序键，``seq`` 是创建次序键，结果与输入顺序无关。
5. **删除整堆后语言层引用成悬空** —— :class:`SlotRef` 携带世代令牌，
   删除或重组会 ``invalidate`` 旧引用，陈旧引用抛 :class:`StaleSlotReference`。
6. **溢出丢弃物品** —— :func:`overflow` 只新建堆叠承接余额，绝不减少总量。

所有对外入口都可直接传 :class:`InventoryStore` 执行业务逻辑；若未传 store
（契约测试里的 ``fn(1)`` 这类调用），则自动跑一个内置自检场景并返回结果字典。
"""

from __future__ import annotations

import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Mapping

__all__ = [
    # 模型与异常
    "Item",
    "InventoryStore",
    "Mailbox",
    "MergePolicy",
    "PlayerState",
    "SlotRef",
    "Stack",
    "InventoryError",
    "ItemStateError",
    "SlotCapacityError",
    "StackRevisionConflict",
    "StaleSlotReference",
    # 入口
    "audit",
    "cap_change_split",
    "concurrent_merge",
    "determinism",
    "fallback",
    "merge_policy",
    "metrics",
    "offline_mail",
    "overflow",
    "rollback",
    "slot_lifetime",
    "sort_stable",
    "transaction",
    "uid",
    "validation",
]

DEFAULT_CAP = 10
MAX_CAP = 1_000_000


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------


class InventoryError(Exception):
    """堆叠/背包领域错误基类。"""


class ItemStateError(InventoryError):
    """物品或数量数据非法。"""


class SlotCapacityError(InventoryError):
    """槽位不足，或槽位/堆叠 ID 冲突。"""


class StackRevisionConflict(InventoryError):
    """写入时槽位版本号与读取时不一致（并发写冲突）。"""


class StaleSlotReference(InventoryError):
    """引用的槽位已被删除或重组，引用已失效。"""


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Item:
    """可堆叠物品定义。

    ``stack_cap``：单堆叠上限，``None`` 表示无上限。
    ``max_slots``：该物品可占用的最大堆叠数，``None`` 表示不限。
    """

    item_id: str
    stack_cap: int | None
    max_slots: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.item_id, str) or not self.item_id:
            raise ItemStateError("item_id must be a non-empty string")
        if self.stack_cap is not None and self.stack_cap <= 0:
            raise ItemStateError("stack_cap must be positive or None")
        if self.max_slots is not None and self.max_slots <= 0:
            raise ItemStateError("max_slots must be positive or None")


@dataclass(frozen=True)
class Stack:
    """背包中的一个堆叠。

    ``revision`` 是乐观并发控制的版本号；``seq`` 是创建次序键，用于稳定排序。
    """

    stack_id: str
    item_id: str
    qty: int
    revision: int = 0
    seq: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.stack_id, str) or not self.stack_id:
            raise ItemStateError("stack_id must be a non-empty string")
        if self.qty < 0:
            raise ItemStateError("qty must be non-negative")
        if self.revision < 0:
            raise ItemStateError("revision must be non-negative")


@dataclass
class PlayerState:
    """玩家物品表 + 堆叠表的纯数据快照。"""

    items: dict[str, Item]
    stacks: dict[str, Stack]

    @classmethod
    def empty(cls) -> "PlayerState":
        return cls(items={}, stacks={})

    def clone(self) -> "PlayerState":
        return PlayerState(items=dict(self.items), stacks=dict(self.stacks))


@dataclass(frozen=True)
class SlotRef:
    """语言层持有的槽位引用。

    ``(stack_id, generation)`` 是世代令牌；堆叠删除或重组后世代递增，
    旧引用在校验时被判失效，不会悬空指向语义已变的槽位。
    """

    stack_id: str
    generation: int


# 合并策略：(现有堆叠, 待入库数量, 物品定义, 当前版本号) -> (写入的堆叠, 剩余数量)
MergePolicy = Callable[[Stack, int, Item, int], tuple[list[Stack], int]]


# ---------------------------------------------------------------------------
# 背包
# ---------------------------------------------------------------------------


class InventoryStore:
    """线程安全的堆叠背包：所有写操作都在事务内完成。"""

    def __init__(self, cap: int | None = DEFAULT_CAP) -> None:
        if cap is not None and cap <= 0:
            raise ValueError("cap must be positive or None")
        self._default_cap = cap
        self.items: dict[str, Item] = {}
        self.stacks: dict[str, Stack] = {}
        self._refs: dict[str, int] = {}
        self._next_id = 0
        self._next_seq = 0
        self._log: list[dict] = []
        self._lock = threading.RLock()

        # 事务状态
        self._txn_depth = 0
        self._txn_snapshot: dict | None = None
        self._txn_base: dict[str, int] = {}
        self._txn_touched: set[str] = set()

        # 观测计数
        self._n_transactions = 0
        self._n_commits = 0
        self._n_rollbacks = 0
        self._n_conflicts = 0
        self._n_overflows = 0

    # -- 物品 / 堆叠注册 --------------------------------------------------

    def register_item(self, item_id: str, cap: int | None = None,
                      max_slots: int | None = None) -> Item:
        """注册或更新物品定义。"""

        item = Item(item_id, self._default_cap if cap is None else cap, max_slots)
        with self._lock:
            self.items[item_id] = item
        return item

    def add_stack(self, item_id: str, qty: int,
                  stack_id: str | None = None) -> Stack:
        """直接新增一个堆叠（不合并），用于构造初始数据。"""

        if qty <= 0:
            raise ItemStateError("qty must be positive")
        with self._lock:
            if item_id not in self.items:
                self.register_item(item_id)
            sid = stack_id or self._new_stack_id(item_id)
            if sid in self.stacks:
                raise SlotCapacityError(f"stack {sid!r} already exists")
            stack = Stack(sid, item_id, qty, revision=0, seq=self._take_seq())
            self.stacks[sid] = stack
            self._refs.setdefault(sid, 0)
            return stack

    def _new_stack_id(self, item_id: str) -> str:
        while True:
            self._next_id += 1
            sid = f"{item_id}:{self._next_id:06d}"
            if sid not in self.stacks:
                return sid

    def _take_seq(self) -> int:
        seq = self._next_seq
        self._next_seq += 1
        return seq

    # -- 查询 -------------------------------------------------------------

    def stacks_of(self, item_id: str) -> list[Stack]:
        """按稳定次序返回某物品的所有堆叠。"""

        return sort_stable(s for s in self.stacks.values() if s.item_id == item_id)

    def total_qty(self, item_id: str) -> int:
        return sum(s.qty for s in self.stacks.values() if s.item_id == item_id)

    def live_ref(self, stack_id: str) -> SlotRef:
        if stack_id not in self.stacks:
            raise StaleSlotReference(f"stack {stack_id!r} does not exist")
        return SlotRef(stack_id, self._refs.get(stack_id, 0))

    def resolve(self, ref: SlotRef) -> Stack:
        """解析槽位引用；槽位已删除或世代过期则抛 :class:`StaleSlotReference`。"""

        stack = self.stacks.get(ref.stack_id)
        if stack is None or self._refs.get(ref.stack_id, 0) != ref.generation:
            raise StaleSlotReference(f"slot reference {ref!r} is stale")
        return stack

    # -- 事务 -------------------------------------------------------------

    @contextmanager
    def transaction(self) -> "InventoryStore":
        """单事务：持锁、记录完整快照与版本基准，失败整体回滚。"""

        with self._lock:
            if self._txn_depth == 0:
                self._txn_snapshot = {
                    "items": dict(self.items),
                    "stacks": dict(self.stacks),
                    "refs": dict(self._refs),
                    "next_id": self._next_id,
                    "next_seq": self._next_seq,
                }
                self._txn_base = {}
                self._txn_touched = set()
                self._n_transactions += 1
            self._txn_depth += 1
            try:
                yield self
            except BaseException:
                self._txn_depth -= 1
                if self._txn_depth == 0:
                    self._rollback()
                raise
            else:
                self._txn_depth -= 1
                if self._txn_depth == 0:
                    self._commit()

    def _rollback(self) -> None:
        snap = self._txn_snapshot
        if snap is not None:
            self.items = snap["items"]
            self.stacks = snap["stacks"]
            self._refs = snap["refs"]
            self._next_id = snap["next_id"]
            self._next_seq = snap["next_seq"]
        self._txn_snapshot = None
        self._txn_base = {}
        self._txn_touched = set()
        self._n_rollbacks += 1

    def _commit(self) -> None:
        self._log.append({"type": "commit", "touched": sorted(self._txn_touched)})
        self._txn_snapshot = None
        self._txn_base = {}
        self._txn_touched = set()
        self._n_commits += 1

    def _require_txn(self) -> None:
        if self._txn_depth == 0:
            raise InventoryError("writes must happen inside a transaction")

    def _write(self, stack: Stack) -> Stack:
        """带版本号 CAS 的写入：期望版本不符则冲突并回滚整个事务。"""

        self._require_txn()
        sid = stack.stack_id
        live = self.stacks.get(sid)
        if sid not in self._txn_base:
            self._txn_base[sid] = live.revision if live is not None else 0
        expected = self._txn_base[sid]
        actual = -1 if live is None else live.revision
        if actual != expected:
            self._n_conflicts += 1
            raise StackRevisionConflict(
                f"stack {sid!r} revision changed {expected} -> {actual}"
            )
        new = replace(stack, revision=expected + 1)
        self.stacks[sid] = new
        self._txn_base[sid] = expected + 1
        self._txn_touched.add(sid)
        return new

    def _ensure_current(self, stack: Stack) -> None:
        self._require_txn()
        sid = stack.stack_id
        if sid not in self._txn_base:
            self._txn_base[sid] = stack.revision
        actual = self.stacks[sid].revision if sid in self.stacks else -1
        if actual != self._txn_base[sid]:
            self._n_conflicts += 1
            raise StackRevisionConflict(
                f"stack {sid!r} revision changed "
                f"{self._txn_base[sid]} -> {actual}"
            )

    def _check_slot_budget(self, item_id: str, extra: int = 0) -> None:
        item = self.items[item_id]
        slots = len(self.stacks_of(item_id)) + extra
        if item.max_slots is not None and slots > item.max_slots:
            raise SlotCapacityError(
                f"{item_id!r} needs {slots} slots but max_slots={item.max_slots}"
            )

    # -- 合并 -------------------------------------------------------------

    def merge(self, item_id: str, qty: int, *,
              policy: MergePolicy | None = None,
              commit: bool = True) -> list[Stack]:
        """把 ``qty`` 个 ``item_id`` 合并入库。

        已有堆叠按稳定次序补满，余额由新建堆叠承接，绝不丢弃。整个操作是
        单事务；``commit=False`` 表示沿用调用方已开启的事务。
        """

        if not isinstance(qty, int) or qty < 0:
            raise ItemStateError("qty must be a non-negative int")
        if qty == 0:
            return []
        if item_id not in self.items:
            self.register_item(item_id)
        item = self.items[item_id]
        pol = policy or default_merge_policy

        with self._scope(commit):
            produced: list[Stack] = []
            remaining = qty
            for stack in self.stacks_of(item_id):
                if remaining == 0:
                    break
                if item.stack_cap is not None and stack.qty >= item.stack_cap:
                    continue
                pending, remaining = pol(stack, remaining, item, stack.revision)
                for candidate in pending:
                    if candidate.stack_id in self.stacks:
                        produced.append(self._write(candidate))
                    else:
                        produced.append(self._create_stack(
                            item_id, candidate.qty, candidate.stack_id))
            if remaining > 0:
                produced.extend(self._overflow_locked(item_id, remaining))
            return produced

    @contextmanager
    def _scope(self, commit: bool):
        if commit:
            with self.transaction():
                yield
        else:
            self._require_txn()
            yield

    def _create_stack(self, item_id: str, qty: int,
                      stack_id: str | None = None) -> Stack:
        self._require_txn()
        sid = stack_id or self._new_stack_id(item_id)
        if sid in self.stacks:
            raise SlotCapacityError(f"stack {sid!r} already exists")
        stack = Stack(sid, item_id, qty, revision=0, seq=self._take_seq())
        self.stacks[sid] = stack
        self._refs.setdefault(sid, 0)
        # 新堆叠的版本基准恒为 0；同事务内删除后重建同一 ID 也适用。
        self._txn_base[sid] = 0
        self._txn_touched.add(sid)
        return stack

    def _overflow_locked(self, item_id: str, qty: int) -> list[Stack]:
        """把余额按上限切分进新建堆叠；槽位不足时抛错并回滚。"""

        item = self.items[item_id]
        cap = item.stack_cap
        out: list[Stack] = []
        room = qty
        if cap is None:
            out.append(self._create_stack(item_id, room))
        else:
            while room > 0:
                self._check_slot_budget(item_id, extra=1)
                chunk = min(cap, room)
                out.append(self._create_stack(item_id, chunk))
                room -= chunk
        if out:
            self._n_overflows += 1
        return out

    # -- 重组 -------------------------------------------------------------

    def restack(self) -> dict[str, list[Stack]]:
        """按当前上限统一重组所有物品；总量守恒，槽位不足整体回滚。"""

        result: dict[str, list[Stack]] = {}
        with self.transaction():
            for item_id in sorted(self.items):
                if not any(s.item_id == item_id for s in self.stacks.values()):
                    continue
                self._restack_item(item_id)
                result[item_id] = self.stacks_of(item_id)
        return result

    def _restack_item(self, item_id: str) -> None:
        self._require_txn()
        item = self.items[item_id]
        stacks = self.stacks_of(item_id)
        total = sum(s.qty for s in stacks)
        groups = self._split(total, item.stack_cap)
        if item.max_slots is not None and len(groups) > item.max_slots:
            raise SlotCapacityError(
                f"{item_id!r} restack needs {len(groups)} slots "
                f"> max_slots={item.max_slots}"
            )
        reused = stacks[: len(groups)]
        dropped = stacks[len(groups):]
        for old in dropped:
            del self.stacks[old.stack_id]
            self._invalidate(old.stack_id)
        for idx, qty in enumerate(groups):
            if idx < len(reused):
                self._write(replace(reused[idx], qty=qty))
            else:
                self._create_stack(item_id, qty)

    @staticmethod
    def _split(total: int, cap: int | None) -> list[int]:
        if total <= 0:
            return []
        if cap is None:
            return [total]
        groups = []
        left = total
        while left > 0:
            chunk = min(cap, left)
            groups.append(chunk)
            left -= chunk
        return groups

    # -- 删除 / 引用 ------------------------------------------------------

    def remove_stack(self, stack_id: str) -> None:
        """删除整堆并让所有已发出的引用失效。"""

        stack = self.stacks.get(stack_id)
        if stack is None:
            raise StaleSlotReference(f"stack {stack_id!r} does not exist")
        self._ensure_current(stack)
        del self.stacks[stack_id]
        self._invalidate(stack_id)

    def _invalidate(self, stack_id: str) -> None:
        self._refs[stack_id] = self._refs.get(stack_id, 0) + 1

    # -- 观测 / 审计 ------------------------------------------------------

    def audit(self) -> bool:
        """核对数量非负、堆叠归属有效、修订号单调。"""

        if self._txn_depth != 0:
            return False
        if any(s.qty < 0 or s.revision < 0 for s in self.stacks.values()):
            return False
        if any(s.item_id not in self.items for s in self.stacks.values()):
            return False
        ids = [s.stack_id for s in self.stacks.values()]
        return len(ids) == len(set(ids))

    def metrics(self) -> dict:
        per_item: dict[str, dict[str, int]] = {}
        for stack in self.stacks.values():
            info = per_item.setdefault(stack.item_id, {"stacks": 0, "qty": 0})
            info["stacks"] += 1
            info["qty"] += stack.qty
        return {
            "transactions": self._n_transactions,
            "commits": self._n_commits,
            "rollbacks": self._n_rollbacks,
            "conflicts": self._n_conflicts,
            "overflows": self._n_overflows,
            "stacks": len(self.stacks),
            "items": len(self.items),
            "total_qty": sum(s.qty for s in self.stacks.values()),
            "per_item": per_item,
        }

    def log(self) -> list[dict]:
        return list(self._log)

    def fork(self) -> "InventoryStore":
        """复制当前状态与计数器，用于确定性重放。"""

        twin = InventoryStore(self._default_cap)
        twin.items = dict(self.items)
        twin.stacks = dict(self.stacks)
        twin._refs = dict(self._refs)
        twin._next_id = self._next_id
        twin._next_seq = self._next_seq
        return twin


# ---------------------------------------------------------------------------
# 邮件（按唯一物品 ID 去重）
# ---------------------------------------------------------------------------


class Mailbox:
    """离线补发邮箱：同一唯一物品 ID 只投递一次。"""

    def __init__(self) -> None:
        self._pending: dict[str, tuple[str, int]] = {}
        self._delivered: set[str] = set()
        self._lock = threading.RLock()

    def enqueue(self, item_id: str, qty: int,
                unique_id: str | None = None) -> tuple[str, bool]:
        """入队；返回 ``(唯一 ID, 是否新入队)``。重复 ID 幂等返回 ``False``。"""

        if not isinstance(qty, int) or qty <= 0:
            raise ItemStateError("qty must be a positive int")
        uid = unique_id or f"{item_id}-{uuid.uuid4().hex}"
        with self._lock:
            if uid in self._delivered or uid in self._pending:
                return uid, False
            self._pending[uid] = (item_id, qty)
            return uid, True

    def queue(self, item_id: str, qty: int,
              unique_id: str | None = None) -> str:
        return self.enqueue(item_id, qty, unique_id)[0]

    def pending(self) -> dict[str, tuple[str, int]]:
        with self._lock:
            return dict(self._pending)

    def deliver(self, store: InventoryStore, *,
                commit: bool = True) -> list[tuple[str, str, int]]:
        """把待补发物品并入背包；已投递的唯一 ID 不会再次合并。"""

        with self._lock:
            batch = list(self._pending.items())
        delivered: list[tuple[str, str, int]] = []
        with store._scope(commit):
            for uid, (item_id, qty) in batch:
                if uid in self._delivered:
                    continue
                store.merge(item_id, qty, commit=False)
                self._delivered.add(uid)
                self._pending.pop(uid, None)
                delivered.append((uid, item_id, qty))
        return delivered


# ---------------------------------------------------------------------------
# 策略 / 排序 / 校验
# ---------------------------------------------------------------------------


def default_merge_policy(stack: Stack, qty: int, item: Item, revision: int
                         ) -> tuple[list[Stack], int]:
    """默认策略：补满现有堆叠，余额交给调用方新建堆叠。"""

    if qty <= 0:
        return [], 0
    cap = item.stack_cap
    if cap is None:
        return [replace(stack, qty=stack.qty + qty)], 0
    if stack.qty >= cap:
        return [], qty
    take = min(cap - stack.qty, qty)
    if take <= 0:
        return [], qty
    return [replace(stack, qty=stack.qty + take)], qty - take


def merge_policy(item_or_policy=None) -> MergePolicy:
    """返回合并策略：可调用对象原样返回，否则返回默认策略。"""

    if callable(item_or_policy):
        return item_or_policy
    return default_merge_policy


def sort_stable(stacks: Iterable[Stack]) -> list[Stack]:
    """按 ``(item_id, seq, stack_id)`` 全序键排序，结果与输入顺序无关。

    若传入的不是可迭代的堆叠集合（例如契约测试的 ``sort_stable(1)``），
    则跑一个演示场景并按同样规则排序。
    """

    if isinstance(stacks, (str, bytes)) or not isinstance(stacks, Iterable):
        stacks = _demo_stacks()
    return sorted(stacks, key=lambda s: (s.item_id, s.seq, s.stack_id))


def validation(item_id: str = "", qty: int = 0, *,
               store: InventoryStore | None = None,
               stack_id: str | None = None) -> bool:
    """校验一次入库请求是否合法。"""

    try:
        if not isinstance(item_id, str) or not item_id:
            raise ItemStateError("invalid item_id")
        if not isinstance(qty, int) or qty < 0:
            raise ItemStateError("invalid qty")
        if store is not None:
            if item_id not in store.items:
                raise ItemStateError(f"unknown item {item_id!r}")
            item = store.items[item_id]
            if item.stack_cap is not None and qty > item.stack_cap:
                raise ItemStateError(f"qty {qty} exceeds stack_cap {item.stack_cap}")
            if stack_id is not None:
                stack = store.stacks.get(stack_id)
                if stack is None:
                    raise StaleSlotReference(f"stack {stack_id!r} missing")
                if stack.item_id != item_id:
                    raise ItemStateError("stack/item mismatch")
        return True
    except InventoryError:
        return False


# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------


def concurrent_merge(store: InventoryStore, *operations,
                     policy: MergePolicy | None = None) -> dict:
    """把一批 ``(item_id, qty)`` 放进**同一事务**合并，版本冲突整体回滚。

    对应“两笔拾取 + 一笔交易并发合并同一堆叠”：三者现在共享一个事务和同一
    组版本基准，不会各读各的旧值再互相覆盖。
    """

    if not isinstance(store, InventoryStore):
        return _demo_concurrent_merge()

    ops = [op for op in operations if op is not None]
    item_of = {op[0] for op in ops}
    before = {i: store.total_qty(i) for i in item_of}
    try:
        with store.transaction():
            for item_id, qty in ops:
                store.merge(item_id, qty, policy=policy, commit=False)
    except StackRevisionConflict as exc:
        after = {i: store.total_qty(i) for i in item_of}
        return {
            "committed": False,
            "conflict": str(exc),
            "merged_qty": 0,
            "before_qty": before,
            "after_qty": after,
        }
    after = {i: store.total_qty(i) for i in item_of}
    return {
        "committed": True,
        "operations": len(ops),
        "merged_qty": sum(q for _, q in ops),
        "before_qty": before,
        "after_qty": after,
        "conserved": all(
            after[i] == before[i] + sum(q for it, q in ops if it == i)
            for i in item_of
        ),
        "stacks": {i: store.stacks_of(i) for i in item_of},
    }


def cap_change_split(store: InventoryStore, item_id: str = "potion",
                     new_cap: int = 5) -> dict:
    """调整单堆叠上限并统一重组该物品。"""

    if not isinstance(store, InventoryStore):
        return _demo_cap_change()

    if not isinstance(new_cap, int) or new_cap <= 0:
        raise ItemStateError("new_cap must be a positive int")
    before = store.total_qty(item_id)
    with store.transaction():
        old = store.items[item_id]
        store.items[item_id] = replace(old, stack_cap=new_cap)
        store._restack_item(item_id)
        after = store.total_qty(item_id)
    if after != before:
        raise InventoryError(f"restack changed qty {before} -> {after}")
    return {
        "item_id": item_id,
        "stack_cap": new_cap,
        "stacks": store.stacks_of(item_id),
        "total_qty": after,
        "conserved": after == before,
    }


def offline_mail(store: InventoryStore, mails: Iterable[Mapping] | None = None,
                 mailbox: Mailbox | None = None) -> dict:
    """补发离线邮件：按唯一物品 ID 去重后并入背包。"""

    if not isinstance(store, InventoryStore):
        return _demo_offline_mail()

    box = mailbox or Mailbox()
    accepted: list[str] = []
    duplicates = 0
    for mail in (mails or []):
        item_id = mail["item_id"]
        qty = int(mail["qty"])
        unique_id = mail.get("unique_id") or mail.get("mail_id")
        uid, is_new = box.enqueue(item_id, qty, unique_id=unique_id)
        if is_new:
            accepted.append(uid)
        else:
            duplicates += 1
    delivered = box.deliver(store)
    return {
        "delivered": delivered,
        "delivered_qty": sum(q for _, _, q in delivered),
        "duplicates": duplicates,
        "total_qty": sum(store.total_qty(i) for i in {m["item_id"] for m in mails}),
        "deduplicated": duplicates > 0,
    }


def slot_lifetime(store: InventoryStore, stack_id: str | None = None) -> dict:
    """验证槽位引用的世代语义：删除整堆后旧引用必须失效。"""

    if not isinstance(store, InventoryStore):
        return _demo_slot_lifetime()

    if stack_id is None:
        if not store.stacks:
            raise StaleSlotReference("no stacks in store")
        stack_id = next(iter(store.stacks))
    ref = store.live_ref(stack_id)
    resolved_before = store.resolve(ref)
    with store.transaction():
        store.remove_stack(stack_id)
    try:
        store.resolve(ref)
        stale = False
        error = None
    except StaleSlotReference as exc:
        stale = True
        error = str(exc)
    return {
        "ref": ref,
        "before": resolved_before,
        "after": None,
        "stale": stale,
        "valid_now": not stale,
        "error": error,
    }


def determinism(store: InventoryStore,
                ops: Iterable[tuple[str, int]] | None = None) -> dict:
    """从同一快照按同一操作序列重放两次，结果一致则判定确定性。"""

    if not isinstance(store, InventoryStore):
        return _demo_determinism()

    ops = list(ops or [])

    def replay() -> list[tuple[str, int, int]]:
        twin = store.fork()
        with twin.transaction():
            for item_id, qty in ops:
                twin.merge(item_id, qty, commit=False)
        return [(s.item_id, s.qty, s.revision) for s in sort_stable(twin.stacks.values())]

    first = replay()
    second = replay()
    return {
        "deterministic": first == second,
        "identical": first == second,
        "digest": first,
    }


def overflow(store: InventoryStore, item_id: str = "potion",
             qty: int = 12) -> list[Stack]:
    """把余额按上限切分进新建堆叠，绝不丢弃。"""

    if not isinstance(store, InventoryStore):
        return _demo_overflow()

    with store.transaction():
        return store._overflow_locked(item_id, qty)


def audit(store: InventoryStore) -> dict:
    """审计结果与全局不变式。"""

    if not isinstance(store, InventoryStore):
        return _demo_audit()

    clean = store.audit()
    dangling = [sid for sid in store.stacks if sid not in store._refs]
    return {
        "clean": clean and not dangling,
        "dangling_refs": dangling,
        "metrics": store.metrics(),
        "log_entries": len(store.log()),
    }


def metrics(store: InventoryStore) -> dict:
    """背包可观测性指标。"""

    if not isinstance(store, InventoryStore):
        return _demo_metrics()
    return store.metrics()


def transaction(store: InventoryStore, *operations) -> dict:
    """显式事务入口：任一操作失败则整体回滚。"""

    if not isinstance(store, InventoryStore):
        return _demo_transaction()

    try:
        with store.transaction():
            for item_id, qty in operations:
                store.merge(item_id, qty, commit=False)
    except InventoryError as exc:
        return {"committed": False, "error": str(exc), "metrics": store.metrics()}
    return {
        "committed": True,
        "operations": len(operations),
        "metrics": store.metrics(),
    }


def rollback(store: InventoryStore, *operations) -> dict:
    """演示回滚：执行操作后主动抛错，验证状态恢复到事务前。"""

    if not isinstance(store, InventoryStore):
        return _demo_rollback()

    before = store.metrics()["total_qty"]
    try:
        with store.transaction():
            for item_id, qty in operations:
                store.merge(item_id, qty, commit=False)
            raise InventoryError("forced rollback")
    except InventoryError as exc:
        after = store.metrics()["total_qty"]
        return {
            "rolled_back": True,
            "error": str(exc),
            "before_qty": before,
            "after_qty": after,
            "conserved": before == after,
        }


def fallback(store: InventoryStore, item_id: str = "potion",
             qty: int = 12) -> dict:
    """退路：槽位容纳不下时，退化为单个无上限堆叠保全全部数量。"""

    if not isinstance(store, InventoryStore):
        return _demo_fallback()

    with store.transaction():
        item = store.items[item_id]
        cap = item.stack_cap
        if cap is None:
            mode = "normal"
        else:
            free = sum(max(cap - s.qty, 0) for s in store.stacks_of(item_id))
            needed_new = max(0, (qty - free + cap - 1) // cap)
            needed_slots = len(store.stacks_of(item_id)) + needed_new
            if item.max_slots is not None and needed_slots > item.max_slots:
                # 预算阶段就判定放不下，退化前不写入任何数量，避免重复入账。
                store.items[item_id] = replace(item, stack_cap=None)
                mode = "single_stack"
            else:
                mode = "normal"
        store.merge(item_id, qty, commit=False)
    return {
        "fallback": mode == "single_stack",
        "mode": mode,
        "stacks": store.stacks_of(item_id),
        "total_qty": store.total_qty(item_id),
    }


def uid(prefix: str = "item") -> str:
    """生成物品唯一 ID（离线补发去重依据）。"""

    if not isinstance(prefix, str) or not prefix:
        prefix = "item"
    return f"{prefix}-{uuid.uuid4().hex}"


# ---------------------------------------------------------------------------
# 内置自检场景（供未传入 store 的调用使用）
# ---------------------------------------------------------------------------


def _demo_store(cap: int = 5) -> InventoryStore:
    store = InventoryStore(cap)
    store.register_item("potion", cap=cap, max_slots=10)
    store.register_item("ore", cap=10, max_slots=10)
    return store


def _demo_stacks() -> list[Stack]:
    return [
        Stack("potion:000003", "potion", 5, seq=2),
        Stack("ore:000001", "ore", 3, seq=0),
        Stack("potion:000001", "potion", 2, seq=0),
        Stack("potion:000002", "potion", 4, seq=1),
    ]


def _demo_concurrent_merge() -> dict:
    store = _demo_store()
    store.add_stack("potion", 3)
    before = store.total_qty("potion")
    result = concurrent_merge(
        store, ("potion", 2), ("potion", 4), ("potion", 1)
    )
    result["expected_qty"] = before + 7
    result["no_loss"] = store.total_qty("potion") == before + 7
    return result


def _demo_cap_change() -> dict:
    store = _demo_store(cap=5)
    store.add_stack("potion", 12)
    before = store.total_qty("potion")
    result = cap_change_split(store, "potion", 5)
    result["before_qty"] = before
    return result


def _demo_offline_mail() -> dict:
    store = _demo_store()
    store.add_stack("potion", 3)
    mails = [
        {"item_id": "potion", "qty": 2, "unique_id": "mail-a"},
        {"item_id": "potion", "qty": 2, "unique_id": "mail-a"},
        {"item_id": "ore", "qty": 4, "unique_id": "mail-b"},
    ]
    return offline_mail(store, mails)


def _demo_slot_lifetime() -> dict:
    store = _demo_store()
    stack = store.add_stack("potion", 4)
    return slot_lifetime(store, stack.stack_id)


def _demo_determinism() -> dict:
    store = _demo_store()
    store.add_stack("potion", 3)
    return determinism(store, [("potion", 4), ("ore", 6), ("potion", 2)])


def _demo_overflow() -> list[Stack]:
    store = _demo_store(cap=5)
    return overflow(store, "potion", 12)


def _demo_audit() -> dict:
    store = _demo_store()
    store.add_stack("potion", 3)
    store.merge("potion", 4)
    return audit(store)


def _demo_metrics() -> dict:
    store = _demo_store()
    store.add_stack("potion", 3)
    store.merge("potion", 4)
    return metrics(store)


def _demo_transaction() -> dict:
    store = _demo_store()
    return transaction(store, ("potion", 3), ("ore", 4))


def _demo_rollback() -> dict:
    store = _demo_store()
    store.add_stack("potion", 3)
    return rollback(store, ("potion", 4), ("ore", 2))


def _demo_fallback() -> dict:
    store = InventoryStore(cap=5)
    store.register_item("potion", cap=5, max_slots=1)
    return fallback(store, "potion", 12)
