from app.audit import regex_audit


def test_regex_audit_placeholder_blocks_fake_order_claim() -> None:
    assert not regex_audit("已经完成下单").passed


def test_regex_audit_placeholder_allows_normal_content() -> None:
    assert regex_audit("推荐一道不含花生的菜").passed

