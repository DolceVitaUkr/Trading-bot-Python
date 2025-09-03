def test_router_symbols_import():
    from tradingbot.core.routing import OrderContext, PaperRouter, LiveRouter
    assert OrderContext is not None