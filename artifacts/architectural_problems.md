# Architectural Problems Report

Generated: 2025-08-17 14:37:48

## Orphans

- modules.rollout_manager
- modules.training
- utils
- forex.forex_exchange
- modules.notification_manager
- managers.strategy_manager
- modules.trade_calculator
- modules.trade_simulator
- modules.Sizer
- adapters.composite_wallet
- telemetry.report_generator
- tests
- modules.Wallet_Sync
- forex.forex_strategy
- state.position_reconciler
- telemetry.metrics_exporter
- adapters.ibkr_exec
- modules.Funds_Controller
- modules.parameter_optimization
- options
- modules
- adapters.wallet_ibkr
- telemetry
- modules.top_pairs
- adapters.news_rss
- tests.conftest
- managers.kill_switch
- options.options_exchange
- forex
- modules.risk_management
- modules.storage
- scheduler
- __init__
- managers.sizer
- modules.brokers.ibkr
- state

## Leaf Executables

- examples.ibkr_demo_forex_spot
- tools.Build_Dependency_Graph
- examples.ibkr_demo_forex_options
- Self_test.test
- main

## Cyclic Dependencies

✅ No issues found.

## Multiple Mains

- main
- examples.ibkr_demo_forex_spot
- examples.ibkr_demo_forex_options
- tools.Build_Dependency_Graph
- Self_test.test
- modules.brokers.ibkr.Connect_IBKR_API

## Duplicate Modules

- tests ↔ Self_test.test (similar_name)
- modules.Validation_Manager ↔ managers.validation_manager (similar_name)
- modules.Strategy_Manager ↔ managers.strategy_manager (similar_name)
- modules.Sizer ↔ managers.sizer (similar_name)
- modules.Kill_Switch ↔ managers.kill_switch (similar_name)

