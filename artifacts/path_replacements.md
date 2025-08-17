# Path Replacement Suggestions

Generated: 2025-08-17 14:39:22

## Summary

- Total hardcoded paths found: 22
- Total direct I/O operations found: 68

## Recommended Data_Registry Methods

Replace hardcoded paths with these Data_Registry method calls:

- `Data_Registry.get_data_path(branch, mode, dataset_type)` - Historical OHLCV data, indicators, features
- `Data_Registry.get_model_path(branch, model_type)` - Trained models, artifacts, checkpoints
- `Data_Registry.get_log_path(branch, mode, log_type)` - Structured logs, error logs, decision traces
- `Data_Registry.get_backup_path(exchange, symbol)` - Historical data backups by exchange
- `Data_Registry.get_state_path(branch, mode)` - Runtime state, positions, metrics
- `Data_Registry.get_metrics_path(branch, mode)` - Performance metrics, backtesting results
- `Data_Registry.get_decisions_path(branch, mode)` - Trading decisions, signals, analysis

## File-by-File Replacements

### Self_test/test.py

Line 139: Use Data_Manager methods with Data_Registry paths
Line 139: Use Data_Manager methods with Data_Registry paths

### config.py

Line 25: Use Log_Manager for structured logging
Line 68: Use Data_Manager.read_json() / write_json() with Data_Registry paths

### managers/kill_switch.py

Line 11: Use Data_Registry.get_state_path(branch, mode)
Line 11: Replace with Data_Registry method call
Line 31: Use Data_Manager methods with Data_Registry paths
Line 31: Use Data_Manager methods with Data_Registry paths
Line 39: Use Data_Manager methods with Data_Registry paths
Line 39: Use Data_Manager methods with Data_Registry paths
Line 65: Use Data_Manager methods with Data_Registry paths
Line 65: Use Data_Manager methods with Data_Registry paths
Line 66: Use Data_Manager.read_json() / write_json() with Data_Registry paths

### managers/strategy_manager.py

Line 10: Use Data_Registry.get_state_path(branch, mode)
Line 10: Replace with Data_Registry method call
Line 24: Use Data_Manager methods with Data_Registry paths
Line 24: Use Data_Manager methods with Data_Registry paths
Line 32: Use Data_Manager methods with Data_Registry paths
Line 32: Use Data_Manager methods with Data_Registry paths
Line 33: Use Data_Manager.read_json() / write_json() with Data_Registry paths

### managers/validation_manager.py

Line 16: Use Data_Registry.get_state_path(branch, mode)
Line 16: Replace with Data_Registry method call
Line 34: Use Data_Manager methods with Data_Registry paths
Line 34: Use Data_Manager methods with Data_Registry paths
Line 39: Use Data_Manager methods with Data_Registry paths
Line 39: Use Data_Manager methods with Data_Registry paths
Line 47: Use Data_Manager methods with Data_Registry paths
Line 47: Use Data_Manager methods with Data_Registry paths
Line 58: Use Data_Manager methods with Data_Registry paths
Line 58: Use Data_Manager methods with Data_Registry paths
Line 59: Use Data_Manager.read_json() / write_json() with Data_Registry paths

### modules/Logger_Config.py

Line 47: Use Data_Registry.get_log_path(branch, mode, log_type)
Line 47: Replace with Data_Registry method call
Line 74: Use Data_Registry.get_log_path(branch, mode, log_type)
Line 74: Replace with Data_Registry method call

### modules/Sizer.py

Line 37: Use Data_Manager methods with Data_Registry paths
Line 37: Use Data_Manager methods with Data_Registry paths
Line 38: Use Data_Manager.read_json() / write_json() with Data_Registry paths

### modules/brokers/ibkr/Fetch_IBKR_MarketData.py

Line 15: Use Data_Registry.get_data_path(branch, mode, dataset_type)

### modules/data_manager.py

Line 48: Use Data_Registry methods instead of hardcoded .csv paths
Line 49: Use Data_Registry methods instead of hardcoded .json paths
Line 208: Use Data_Manager.read_csv() / write_csv() with Data_Registry paths
Line 229: Use Data_Manager.read_csv() / write_csv() with Data_Registry paths
Line 256: Use Data_Manager methods with Data_Registry paths
Line 256: Use Data_Manager methods with Data_Registry paths
Line 281: Use Data_Manager methods with Data_Registry paths
Line 281: Use Data_Manager methods with Data_Registry paths

### modules/parameter_optimization.py

Line 42: Replace with Data_Registry method call
Line 87: Use Data_Manager methods with Data_Registry paths
Line 87: Use Data_Manager methods with Data_Registry paths
Line 88: Use Save_AI_Update module for model persistence
Line 289: Use Data_Manager methods with Data_Registry paths
Line 289: Use Data_Manager methods with Data_Registry paths
Line 290: Use Save_AI_Update module for model persistence

### modules/storage/Save_AI_Update.py

Line 30: Replace with Data_Registry method call
Line 35: Use Data_Manager methods with Data_Registry paths
Line 35: Use Data_Manager methods with Data_Registry paths
Line 36: Use Save_AI_Update module for model persistence
Line 51: Replace with Data_Registry method call
Line 59: Use Data_Manager methods with Data_Registry paths
Line 59: Use Data_Manager methods with Data_Registry paths
Line 60: Use Save_AI_Update module for model persistence

### modules/telegram_bot.py

Line 62: Use Data_Manager.read_json() / write_json() with Data_Registry paths
Line 79: Use Data_Manager.read_json() / write_json() with Data_Registry paths

### state/runtime_state.py

Line 15: Use Data_Registry methods instead of hardcoded .json paths
Line 22: Use Data_Registry methods instead of hardcoded .json paths
Line 127: Use Data_Manager methods with Data_Registry paths
Line 127: Use Data_Manager methods with Data_Registry paths
Line 128: Use Data_Manager.read_json() / write_json() with Data_Registry paths
Line 158: Use Data_Manager methods with Data_Registry paths
Line 158: Use Data_Manager methods with Data_Registry paths
Line 159: Use Data_Manager.read_json() / write_json() with Data_Registry paths

### telemetry/report_generator.py

Line 62: Use Data_Registry methods instead of hardcoded .csv paths
Line 63: Use Data_Manager.read_csv() / write_csv() with Data_Registry paths

### tools/Build_Dependency_Graph.py

Line 80: Use Data_Manager methods with Data_Registry paths
Line 80: Use Data_Manager methods with Data_Registry paths
Line 283: Use Data_Registry methods instead of hardcoded .json paths
Line 284: Use Data_Manager methods with Data_Registry paths
Line 284: Use Data_Manager methods with Data_Registry paths
Line 285: Use Data_Manager.read_json() / write_json() with Data_Registry paths
Line 317: Use Data_Manager methods with Data_Registry paths
Line 317: Use Data_Manager methods with Data_Registry paths
Line 334: Use Data_Manager methods with Data_Registry paths
Line 334: Use Data_Manager methods with Data_Registry paths
Line 377: Use Data_Registry methods instead of hardcoded .json paths

### utils/utilities.py

Line 37: Use Data_Manager.read_json() / write_json() with Data_Registry paths
Line 40: Use Data_Manager methods with Data_Registry paths
Line 46: Use Data_Manager methods with Data_Registry paths
Line 55: Use Data_Manager methods with Data_Registry paths
Line 56: Use Data_Manager.read_json() / write_json() with Data_Registry paths

