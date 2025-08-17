# Hardcoded Paths and Direct I/O Report

Generated: 2025-08-17 14:39:22

Total findings: 90

## Csv File Extension

Found 2 instances:

**telemetry/report_generator.py:62**
```python
csv_path = os.path.join(out_dir, f"{market}_weekly.csv")
```
**Suggestion:** Use Data_Registry methods instead of hardcoded .csv paths

**modules/data_manager.py:48**
```python
csv_path = os.path.join(base, f"{symfile}.csv")
```
**Suggestion:** Use Data_Registry methods instead of hardcoded .csv paths

## Data Directory Path

Found 1 instances:

**modules/brokers/ibkr/Fetch_IBKR_MarketData.py:15**
```python
CACHE_DIR = Path("data/ibkr/historical")
```
**Suggestion:** Use Data_Registry.get_data_path(branch, mode, dataset_type)

## Direct File Open

Found 26 instances:

**state/runtime_state.py:127**
```python
with open(self.path, "r", encoding="utf-8") as f:
```
**Suggestion:** Use Data_Manager methods with Data_Registry paths

**state/runtime_state.py:158**
```python
with open(tmp, "w", encoding="utf-8") as f:
```
**Suggestion:** Use Data_Manager methods with Data_Registry paths

**tools/Build_Dependency_Graph.py:80**
```python
with open(py_file, 'r', encoding='utf-8') as f:
```
**Suggestion:** Use Data_Manager methods with Data_Registry paths

**tools/Build_Dependency_Graph.py:284**
```python
with open(output_path, 'w') as f:
```
**Suggestion:** Use Data_Manager methods with Data_Registry paths

**tools/Build_Dependency_Graph.py:317**
```python
with open(dot_path, 'w') as f:
```
**Suggestion:** Use Data_Manager methods with Data_Registry paths

**tools/Build_Dependency_Graph.py:334**
```python
with open(output_path, 'w') as f:
```
**Suggestion:** Use Data_Manager methods with Data_Registry paths

**Self_test/test.py:139**
```python
with open(report_path, "w", encoding="utf-8") as f:
```
**Suggestion:** Use Data_Manager methods with Data_Registry paths

**modules/Sizer.py:37**
```python
with open(policy_path, 'r') as f:
```
**Suggestion:** Use Data_Manager methods with Data_Registry paths

**modules/data_manager.py:256**
```python
with open(path, "rb") as f:
```
**Suggestion:** Use Data_Manager methods with Data_Registry paths

**modules/data_manager.py:281**
```python
with open(path, "a", encoding="utf-8") as f:
```
**Suggestion:** Use Data_Manager methods with Data_Registry paths

**modules/parameter_optimization.py:87**
```python
with open(self.checkpoint_file, "rb") as f:
```
**Suggestion:** Use Data_Manager methods with Data_Registry paths

**modules/parameter_optimization.py:289**
```python
with open(self.checkpoint_file, "wb") as f:
```
**Suggestion:** Use Data_Manager methods with Data_Registry paths

**utils/utilities.py:40**
```python
with io.open(tmp_path, "w", encoding="utf-8") as f:
```
**Suggestion:** Use Data_Manager methods with Data_Registry paths

**utils/utilities.py:46**
```python
with io.open(path, "w", encoding="utf-8") as f:
```
**Suggestion:** Use Data_Manager methods with Data_Registry paths

**utils/utilities.py:55**
```python
with io.open(path, "r", encoding="utf-8") as f:
```
**Suggestion:** Use Data_Manager methods with Data_Registry paths

**managers/strategy_manager.py:24**
```python
with open(self.strategy_file, "a"):
```
**Suggestion:** Use Data_Manager methods with Data_Registry paths

**managers/strategy_manager.py:32**
```python
with open(self.strategy_file, "ab") as f:
```
**Suggestion:** Use Data_Manager methods with Data_Registry paths

**managers/validation_manager.py:34**
```python
with open(self.validation_file, "a"):
```
**Suggestion:** Use Data_Manager methods with Data_Registry paths

**managers/validation_manager.py:39**
```python
with open(self.validation_file, "a"):
```
**Suggestion:** Use Data_Manager methods with Data_Registry paths

**managers/validation_manager.py:47**
```python
with open(self.validation_file, "rb") as f:
```
**Suggestion:** Use Data_Manager methods with Data_Registry paths

**managers/validation_manager.py:58**
```python
with open(self.validation_file, "ab") as f:
```
**Suggestion:** Use Data_Manager methods with Data_Registry paths

**managers/kill_switch.py:31**
```python
with open(self.kill_switch_file, "a"):
```
**Suggestion:** Use Data_Manager methods with Data_Registry paths

**managers/kill_switch.py:39**
```python
with open(self.kill_switch_file, "rb") as f:
```
**Suggestion:** Use Data_Manager methods with Data_Registry paths

**managers/kill_switch.py:65**
```python
with open(self.kill_switch_file, "ab") as f:
```
**Suggestion:** Use Data_Manager methods with Data_Registry paths

**modules/storage/Save_AI_Update.py:35**
```python
with open(file_path, "wb") as f:
```
**Suggestion:** Use Data_Manager methods with Data_Registry paths

**modules/storage/Save_AI_Update.py:59**
```python
with open(file_path, "rb") as f:
```
**Suggestion:** Use Data_Manager methods with Data_Registry paths

## Direct File Open Context

Found 23 instances:

**state/runtime_state.py:127**
```python
with open(self.path, "r", encoding="utf-8") as f:
```
**Suggestion:** Use Data_Manager methods with Data_Registry paths

**state/runtime_state.py:158**
```python
with open(tmp, "w", encoding="utf-8") as f:
```
**Suggestion:** Use Data_Manager methods with Data_Registry paths

**tools/Build_Dependency_Graph.py:80**
```python
with open(py_file, 'r', encoding='utf-8') as f:
```
**Suggestion:** Use Data_Manager methods with Data_Registry paths

**tools/Build_Dependency_Graph.py:284**
```python
with open(output_path, 'w') as f:
```
**Suggestion:** Use Data_Manager methods with Data_Registry paths

**tools/Build_Dependency_Graph.py:317**
```python
with open(dot_path, 'w') as f:
```
**Suggestion:** Use Data_Manager methods with Data_Registry paths

**tools/Build_Dependency_Graph.py:334**
```python
with open(output_path, 'w') as f:
```
**Suggestion:** Use Data_Manager methods with Data_Registry paths

**Self_test/test.py:139**
```python
with open(report_path, "w", encoding="utf-8") as f:
```
**Suggestion:** Use Data_Manager methods with Data_Registry paths

**modules/Sizer.py:37**
```python
with open(policy_path, 'r') as f:
```
**Suggestion:** Use Data_Manager methods with Data_Registry paths

**modules/data_manager.py:256**
```python
with open(path, "rb") as f:
```
**Suggestion:** Use Data_Manager methods with Data_Registry paths

**modules/data_manager.py:281**
```python
with open(path, "a", encoding="utf-8") as f:
```
**Suggestion:** Use Data_Manager methods with Data_Registry paths

**modules/parameter_optimization.py:87**
```python
with open(self.checkpoint_file, "rb") as f:
```
**Suggestion:** Use Data_Manager methods with Data_Registry paths

**modules/parameter_optimization.py:289**
```python
with open(self.checkpoint_file, "wb") as f:
```
**Suggestion:** Use Data_Manager methods with Data_Registry paths

**managers/strategy_manager.py:24**
```python
with open(self.strategy_file, "a"):
```
**Suggestion:** Use Data_Manager methods with Data_Registry paths

**managers/strategy_manager.py:32**
```python
with open(self.strategy_file, "ab") as f:
```
**Suggestion:** Use Data_Manager methods with Data_Registry paths

**managers/validation_manager.py:34**
```python
with open(self.validation_file, "a"):
```
**Suggestion:** Use Data_Manager methods with Data_Registry paths

**managers/validation_manager.py:39**
```python
with open(self.validation_file, "a"):
```
**Suggestion:** Use Data_Manager methods with Data_Registry paths

**managers/validation_manager.py:47**
```python
with open(self.validation_file, "rb") as f:
```
**Suggestion:** Use Data_Manager methods with Data_Registry paths

**managers/validation_manager.py:58**
```python
with open(self.validation_file, "ab") as f:
```
**Suggestion:** Use Data_Manager methods with Data_Registry paths

**managers/kill_switch.py:31**
```python
with open(self.kill_switch_file, "a"):
```
**Suggestion:** Use Data_Manager methods with Data_Registry paths

**managers/kill_switch.py:39**
```python
with open(self.kill_switch_file, "rb") as f:
```
**Suggestion:** Use Data_Manager methods with Data_Registry paths

**managers/kill_switch.py:65**
```python
with open(self.kill_switch_file, "ab") as f:
```
**Suggestion:** Use Data_Manager methods with Data_Registry paths

**modules/storage/Save_AI_Update.py:35**
```python
with open(file_path, "wb") as f:
```
**Suggestion:** Use Data_Manager methods with Data_Registry paths

**modules/storage/Save_AI_Update.py:59**
```python
with open(file_path, "rb") as f:
```
**Suggestion:** Use Data_Manager methods with Data_Registry paths

## Json Dump

Found 8 instances:

**state/runtime_state.py:159**
```python
json.dump(self.state, f, indent=2, sort_keys=False)
```
**Suggestion:** Use Data_Manager.read_json() / write_json() with Data_Registry paths

**tools/Build_Dependency_Graph.py:285**
```python
json.dump(self.module_map, f, indent=2, sort_keys=True)
```
**Suggestion:** Use Data_Manager.read_json() / write_json() with Data_Registry paths

**modules/telegram_bot.py:62**
```python
text = json.dumps(text, indent=2)
```
**Suggestion:** Use Data_Manager.read_json() / write_json() with Data_Registry paths

**modules/telegram_bot.py:79**
```python
text = json.dumps(text, indent=2)
```
**Suggestion:** Use Data_Manager.read_json() / write_json() with Data_Registry paths

**utils/utilities.py:37**
```python
payload = json.dumps(data, indent=indent, ensure_ascii=False)
```
**Suggestion:** Use Data_Manager.read_json() / write_json() with Data_Registry paths

**managers/strategy_manager.py:33**
```python
f.write(orjson.dumps(meta.dict(by_alias=True)))
```
**Suggestion:** Use Data_Manager.read_json() / write_json() with Data_Registry paths

**managers/validation_manager.py:59**
```python
f.write(orjson.dumps(record.dict(by_alias=True)))
```
**Suggestion:** Use Data_Manager.read_json() / write_json() with Data_Registry paths

**managers/kill_switch.py:66**
```python
f.write(orjson.dumps(event.dict(by_alias=True)))
```
**Suggestion:** Use Data_Manager.read_json() / write_json() with Data_Registry paths

## Json File Extension

Found 5 instances:

**state/runtime_state.py:15**
```python
os.path.join(STATE_DIR, "runtime.json"))
```
**Suggestion:** Use Data_Registry methods instead of hardcoded .json paths

**state/runtime_state.py:22**
```python
RUNTIME_STATE_FILE = os.path.join(STATE_DIR, "runtime.json")
```
**Suggestion:** Use Data_Registry methods instead of hardcoded .json paths

**tools/Build_Dependency_Graph.py:283**
```python
output_path = self.root_path / "artifacts" / "module_map.json"
```
**Suggestion:** Use Data_Registry methods instead of hardcoded .json paths

**tools/Build_Dependency_Graph.py:377**
```python
print("- module_map.json")
```
**Suggestion:** Use Data_Registry methods instead of hardcoded .json paths

**modules/data_manager.py:49**
```python
meta_path = os.path.join(base, f"{symfile}.meta.json")
```
**Suggestion:** Use Data_Registry methods instead of hardcoded .json paths

## Json Load

Found 4 instances:

**config.py:68**
```python
MD_SUBSCRIPTIONS = orjson.loads(_md_subscriptions_str)
```
**Suggestion:** Use Data_Manager.read_json() / write_json() with Data_Registry paths

**state/runtime_state.py:128**
```python
data = json.load(f)
```
**Suggestion:** Use Data_Manager.read_json() / write_json() with Data_Registry paths

**modules/Sizer.py:38**
```python
policy = json.load(f)
```
**Suggestion:** Use Data_Manager.read_json() / write_json() with Data_Registry paths

**utils/utilities.py:56**
```python
return json.load(f)
```
**Suggestion:** Use Data_Manager.read_json() / write_json() with Data_Registry paths

## Jsonl File Extension

Found 5 instances:

**modules/Logger_Config.py:47**
```python
file_handler = logging.FileHandler("logs/structured_logs.jsonl", mode="a")
```
**Suggestion:** Replace with Data_Registry method call

**modules/Logger_Config.py:74**
```python
log.info("Logging configured", log_level=log_level, file_path="logs/structured_logs.jsonl")
```
**Suggestion:** Replace with Data_Registry method call

**managers/strategy_manager.py:10**
```python
STRATEGY_FILE = "state/strategies.jsonl"
```
**Suggestion:** Replace with Data_Registry method call

**managers/validation_manager.py:16**
```python
VALIDATION_FILE = "state/validation_runs.jsonl"
```
**Suggestion:** Replace with Data_Registry method call

**managers/kill_switch.py:11**
```python
KILL_SWITCH_FILE = "state/kill_switch.jsonl"
```
**Suggestion:** Replace with Data_Registry method call

## Log File Extension

Found 1 instances:

**config.py:25**
```python
LOG_FILE = os.getenv("LOG_FILE", "bot.log")
```
**Suggestion:** Use Log_Manager for structured logging

## Logs Directory Path

Found 2 instances:

**modules/Logger_Config.py:47**
```python
file_handler = logging.FileHandler("logs/structured_logs.jsonl", mode="a")
```
**Suggestion:** Use Data_Registry.get_log_path(branch, mode, log_type)

**modules/Logger_Config.py:74**
```python
log.info("Logging configured", log_level=log_level, file_path="logs/structured_logs.jsonl")
```
**Suggestion:** Use Data_Registry.get_log_path(branch, mode, log_type)

## Pandas Read Csv

Found 1 instances:

**modules/data_manager.py:208**
```python
df = pd.read_csv(path)
```
**Suggestion:** Use Data_Manager.read_csv() / write_csv() with Data_Registry paths

## Pandas To Csv

Found 2 instances:

**telemetry/report_generator.py:63**
```python
df.to_csv(csv_path, index=False)
```
**Suggestion:** Use Data_Manager.read_csv() / write_csv() with Data_Registry paths

**modules/data_manager.py:229**
```python
df.to_csv(path, index=False)
```
**Suggestion:** Use Data_Manager.read_csv() / write_csv() with Data_Registry paths

## Pickle Dump

Found 2 instances:

**modules/parameter_optimization.py:290**
```python
pickle.dump(data, f)
```
**Suggestion:** Use Save_AI_Update module for model persistence

**modules/storage/Save_AI_Update.py:36**
```python
pickle.dump(artifact, f)
```
**Suggestion:** Use Save_AI_Update module for model persistence

## Pickle File Extension

Found 3 instances:

**modules/parameter_optimization.py:42**
```python
"optimization_checkpoint.pkl")
```
**Suggestion:** Replace with Data_Registry method call

**modules/storage/Save_AI_Update.py:30**
```python
file_path = product_dir / f"{strategy_id}.pkl"
```
**Suggestion:** Replace with Data_Registry method call

**modules/storage/Save_AI_Update.py:51**
```python
file_path = BASE_MODEL_DIR / product_name / f"{strategy_id}.pkl"
```
**Suggestion:** Replace with Data_Registry method call

## Pickle Load

Found 2 instances:

**modules/parameter_optimization.py:88**
```python
data = pickle.load(f)
```
**Suggestion:** Use Save_AI_Update module for model persistence

**modules/storage/Save_AI_Update.py:60**
```python
return pickle.load(f)
```
**Suggestion:** Use Save_AI_Update module for model persistence

## State Directory Path

Found 3 instances:

**managers/strategy_manager.py:10**
```python
STRATEGY_FILE = "state/strategies.jsonl"
```
**Suggestion:** Use Data_Registry.get_state_path(branch, mode)

**managers/validation_manager.py:16**
```python
VALIDATION_FILE = "state/validation_runs.jsonl"
```
**Suggestion:** Use Data_Registry.get_state_path(branch, mode)

**managers/kill_switch.py:11**
```python
KILL_SWITCH_FILE = "state/kill_switch.jsonl"
```
**Suggestion:** Use Data_Registry.get_state_path(branch, mode)

