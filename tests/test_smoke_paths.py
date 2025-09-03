def test_import_routing_and_context():
    from tradingbot.core.routing import OrderContext, PaperRouter, LiveRouter
    assert OrderContext is not None and PaperRouter and LiveRouter

def test_import_lifecycles():
    from tradingbot.core.futures_lifecycle import FuturesLifecycle
    from tradingbot.core.options_lifecycle import OptionsLifecycle
    assert FuturesLifecycle and OptionsLifecycle

def test_import_clamp():
    from tradingbot.core.exchange_conformance import clamp_order_if_needed
    assert callable(clamp_order_if_needed)