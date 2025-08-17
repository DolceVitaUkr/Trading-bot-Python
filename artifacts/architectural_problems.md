# Architectural Problems Report

Generated: 2025-08-17 15:00:40

## Orphans

- managers.strategy_manager
- modules.training
- __init__
- modules.notification_manager
- modules.Funds_Controller
- modules
- adapters.news_rss
- state
- adapters.composite_wallet
- telemetry.report_generator
- tests
- modules.top_pairs
- modules.Sizer
- forex.forex_strategy
- managers.kill_switch
- scheduler
- modules.rollout_manager
- utils
- modules.Wallet_Sync
- options.options_exchange
- modules.risk_management
- modules.trade_simulator
- adapters.wallet_ibkr
- adapters.ibkr_exec
- forex.forex_exchange
- modules.brokers.ibkr
- forex
- state.position_reconciler
- modules.storage
- modules.trade_calculator
- managers.sizer
- telemetry
- modules.parameter_optimization
- options
- tests.conftest
- telemetry.metrics_exporter

## Leaf Executables

- Data_Registry
- tools.Find_Hardcoded_Paths
- tools.Build_Dependency_Graph
- examples.ibkr_demo_forex_spot
- tools.Check_Interface_Conformance
- tools.Replace_Print_With_Logger
- Self_test.test
- main
- examples.ibkr_demo_forex_options
- tools.Check_Schemas

## Cyclic Dependencies

✅ No issues found.

## Multiple Mains

- main
- Data_Registry
- examples.ibkr_demo_forex_spot
- examples.ibkr_demo_forex_options
- tools.Find_Hardcoded_Paths
- tools.Check_Schemas
- tools.Check_Interface_Conformance
- tools.Replace_Print_With_Logger
- tools.Build_Dependency_Graph
- Self_test.test
- modules.brokers.ibkr.Connect_IBKR_API

## Duplicate Modules

- tests ↔ Self_test.test (similar_name)
- modules.Validation_Manager ↔ managers.validation_manager (similar_name)
- modules.Strategy_Manager ↔ managers.strategy_manager (similar_name)
- modules.Sizer ↔ managers.sizer (similar_name)
- modules.Kill_Switch ↔ managers.kill_switch (similar_name)

