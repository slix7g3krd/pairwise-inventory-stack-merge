"""堆叠合并修复点的行为测试（补足契约测试只判非空的不足）。"""

import threading

import pytest

import gameplay.inventory_stack_merge as mod
from gameplay.inventory_stack_merge import (
    InventoryStore,
    Mailbox,
    SlotRef,
    StackRevisionConflict,
    StaleSlotReference,
    concurrent_merge,
    cap_change_split,
    offline_mail,
    overflow,
    slot_lifetime,
    sort_stable,
)


def make_store(cap=5):
    store = InventoryStore(cap)
    store.register_item("potion", cap=cap, max_slots=50)
    return store


def test_concurrent_merge_conserves_total():
    store = make_store()
    store.add_stack("potion", 3)
    result = concurrent_merge(store, ("potion", 2), ("potion", 4), ("potion", 1))
    assert result["committed"] is True
    assert store.total_qty("potion") == 10
    assert result["conserved"] is True


def test_parallel_merge_threads_do_not_lose_qty():
    store = make_store(cap=7)

    def worker(n):
        for _ in range(n):
            store.merge("potion", 1)

    threads = [threading.Thread(target=worker, args=(50,)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert store.total_qty("potion") == 200


def test_generated_stacks_never_exceed_cap():
    store = make_store(cap=5)
    overflow(store, "potion", 23)
    assert store.total_qty("potion") == 23
    assert all(s.qty <= 5 for s in store.stacks_of("potion"))


def test_cap_change_restacks_old_stacks():
    store = make_store(cap=10)
    store.add_stack("potion", 23)  # 旧堆叠允许超过后续上限
    result = cap_change_split(store, "potion", 5)
    assert result["conserved"] is True
    assert result["total_qty"] == 23
    assert all(s.qty <= 5 for s in store.stacks_of("potion"))
    assert len(store.stacks_of("potion")) == 5


def test_offline_mail_deduplicates_by_uid():
    store = make_store()
    store.add_stack("potion", 3)
    mails = [
        {"item_id": "potion", "qty": 2, "unique_id": "m1"},
        {"item_id": "potion", "qty": 2, "unique_id": "m1"},  # 同一物品 ID 重放
    ]
    result = offline_mail(store, mails)
    assert result["duplicates"] == 1
    assert store.total_qty("potion") == 5  # 只补发一次


def test_offline_mail_replay_after_delivery_is_noop():
    store = make_store()
    box = Mailbox()
    mails = [{"item_id": "potion", "qty": 4, "unique_id": "m9"}]
    offline_mail(store, mails, box)
    offline_mail(store, mails, box)  # 重登重放
    assert store.total_qty("potion") == 4


def test_sort_is_independent_of_input_order():
    a = mod.Stack("potion:000002", "potion", 1, seq=1)
    b = mod.Stack("potion:000001", "potion", 1, seq=0)
    c = mod.Stack("ore:000001", "ore", 1, seq=2)
    assert sort_stable([a, b, c]) == sort_stable([c, a, b])
    assert [s.stack_id for s in sort_stable([a, b, c])] == [
        "ore:000001",
        "potion:000001",
        "potion:000002",
    ]


def test_slot_reference_invalidated_on_delete():
    store = make_store()
    stack = store.add_stack("potion", 4)
    ref = store.live_ref(stack.stack_id)
    assert store.resolve(ref).qty == 4
    result = slot_lifetime(store, stack.stack_id)
    assert result["stale"] is True
    with pytest.raises(StaleSlotReference):
        store.resolve(ref)


def test_stale_generation_cannot_alias_recreated_slot():
    store = make_store()
    stack = store.add_stack("potion", 4)
    ref = store.live_ref(stack.stack_id)
    with store.transaction():
        store.remove_stack(stack.stack_id)
    store.add_stack("ore", 9, stack_id=stack.stack_id)  # 复用同一槽位 ID
    with pytest.raises(StaleSlotReference):
        store.resolve(ref)


def test_overflow_creates_new_stacks_not_drops():
    store = make_store(cap=5)
    created = overflow(store, "potion", 12)
    assert sum(s.qty for s in created) == 12
    assert len(created) == 3
    assert store.metrics()["overflows"] >= 1


def test_rollback_restores_state():
    store = make_store()
    store.add_stack("potion", 3)
    result = mod.rollback(store, ("potion", 4))
    assert result["rolled_back"] is True
    assert result["conserved"] is True
    assert store.total_qty("potion") == 3


def test_version_conflict_detected_on_stale_write():
    store = make_store()
    stack = store.add_stack("potion", 1)
    with pytest.raises(StackRevisionConflict):
        with store.transaction():
            # 事务读到版本 0 作为基准，随后槽位被并发写推进到 99
            store._txn_base[stack.stack_id] = 0
            store.stacks[stack.stack_id] = mod.replace(
                store.stacks[stack.stack_id], revision=99
            )
            store._write(mod.replace(store.stacks[stack.stack_id], qty=2))
    # 事务回滚后状态复原
    assert store.total_qty("potion") == 1
    assert store.stacks[stack.stack_id].revision == 0


def test_restack_is_transactional_on_slot_shortage():
    store = InventoryStore(cap=5)
    store.register_item("potion", cap=5, max_slots=2)
    store.add_stack("potion", 3)
    store.add_stack("potion", 3)
    with pytest.raises(mod.SlotCapacityError):
        cap_change_split(store, "potion", 1)  # 需要 6 槽 > 2
    assert store.total_qty("potion") == 6  # 回滚，数量不丢


def test_audit_clean_after_normal_ops():
    store = make_store()
    store.merge("potion", 12)
    assert mod.audit(store)["clean"] is True


def test_determinism_replay_matches():
    store = make_store()
    store.add_stack("potion", 3)
    result = mod.determinism(store, [("potion", 4), ("potion", 2)])
    assert result["deterministic"] is True
    assert store.total_qty("potion") == 3  # 重放在副本上，原背包不变


def test_fallback_preserves_qty_when_slots_short():
    store = InventoryStore(cap=5)
    store.register_item("potion", cap=5, max_slots=1)
    result = mod.fallback(store, "potion", 12)
    assert result["fallback"] is True
    assert result["total_qty"] == 12
