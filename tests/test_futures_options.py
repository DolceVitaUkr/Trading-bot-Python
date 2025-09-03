def test_import_lifecycles():
    import tradingbot.core.futures_lifecycle as fl
    import tradingbot.core.options_lifecycle as ol
    assert hasattr(fl, 'FuturesLifecycle')
    assert hasattr(ol, 'OptionsLifecycle')