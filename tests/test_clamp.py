def test_dummy_clamp_import():
    import tradingbot.core.exchange_conformance as ec
    assert hasattr(ec, 'clamp_order_if_needed')