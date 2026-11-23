import pytest
import gameplay.inventory_stack_merge as mod

CASES = [
    ("concurrent_merge", (1,)),
    ("cap_change_split", (1,)),
    ("offline_mail", (1,)),
    ("sort_stable", (1,)),
    ("slot_lifetime", (1,)),
    ("determinism", (1,)),
    ("merge_policy", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("overflow", (1,)),
    ("metrics", (1,)),
    ("transaction", (1,)),
    ("rollback", (1,)),
    ("uid", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
