# File Renames Map for Naming Convention Compliance

## Overview
This document provides the mapping for renaming files to comply with the project standard: **lowercase filenames with no underscores** (except `__init__.py` and `__main__.py`).

---

## Naming Convention Standard
- **Rule**: All filenames should be lowercase with no underscores
- **Exceptions**: `__init__.py`, `__main__.py` (Python special files)
- **Source**: README.md project guidelines

---

## Files Requiring Rename

### Core Module Files

| Current Name | New Name | Status | Impact |
|--------------|----------|--------|---------|
| `config_manager.py` | `configmanager.py` | ✅ **HAS BOTH** - Remove underscore version | Breaking: All imports need update |
| `data_manager.py` | `datamanager.py` | ✅ **HAS BOTH** - Remove underscore version | Breaking: All imports need update |
| `portfolio_manager.py` | `portfoliomanager.py` | ✅ **HAS BOTH** - Remove underscore version | Breaking: All imports need update |
| `risk_manager.py` | `riskmanager.py` | ✅ **HAS BOTH** - Remove underscore version | Breaking: All imports need update |
| `trade_executor.py` | `tradeexecutor.py` | ❌ **ONLY UNDERSCORE** - Needs rename | Breaking: All imports need update |
| `validation_manager.py` | `validationmanager.py` | ❌ **ONLY UNDERSCORE** - Needs rename | Breaking: All imports need update |
| `runtime_controller.py` | `runtimecontroller.py` | ❌ **ONLY UNDERSCORE** - Needs rename | Breaking: All imports need update |
| `error_handler.py` | `errorhandler.py` | ✅ **HAS BOTH** - Remove underscore version | Breaking: All imports need update |

### Broker Module Files

| Current Name | New Name | Status | Impact |
|--------------|----------|--------|---------|
| `exchangebybit.py` | ✅ **COMPLIANT** | No change needed | None |
| `exchangeibkr.py` | ✅ **COMPLIANT** | No change needed | None |
| `exchange.py` | ✅ **COMPLIANT** | No change needed | None |

### Test Files (if any found)

| Current Name | New Name | Status | Impact |
|--------------|----------|--------|---------|
| `test_*.py` | `test*.py` | Would need rename if found | Breaking: Test discovery |

---

## Import Update Strategy

### Phase 1: Remove Duplicate Underscore Files
For files that have BOTH versions (e.g., `config_manager.py` and `configmanager.py`):

1. **Verify** the non-underscore version is complete and functional
2. **Update** all imports to use non-underscore version
3. **Remove** underscore version
4. **Test** that all functionality works

### Phase 2: Rename Underscore-Only Files
For files that only exist with underscores:

1. **Rename** the file to remove underscores
2. **Update** all import statements
3. **Update** any string references to the module name
4. **Test** all functionality

---

## Import Update Locations

### Files That Import `config_manager`:
```python
# Old import (to be removed):
from tradingbot.core.config_manager import ConfigManager

# New import (keep):
from tradingbot.core.configmanager import ConfigManager
```

**Found in**:
- `tradingbot/core/data_manager.py:4`
- `tradingbot/core/portfolio_manager.py:5`
- `tradingbot/core/risk_manager.py:6`
- `tradingbot/core/trade_executor.py:3`
- `tradingbot/brokers/exchangebybit.py:5`
- `tradingbot/ui/app.py:8`

### Files That Import `data_manager`:
```python
# Old import (to be removed):
from tradingbot.core.data_manager import DataManager

# New import (keep):
from tradingbot.core.datamanager import DataManager
```

**Found in**:
- `tradingbot/core/portfolio_manager.py:4`
- `tradingbot/core/trade_executor.py:4`
- `tradingbot/learning/trainmlmodel.py:3`
- `tradingbot/learning/trainrlmodel.py:4`

### Files That Import `risk_manager`:
```python
# Old import (to be removed):
from tradingbot.core.risk_manager import RiskManager

# New import (keep):
from tradingbot.core.riskmanager import RiskManager
```

**Found in**:
- `tradingbot/core/trade_executor.py:5`
- `tradingbot/core/portfolio_manager.py:6`

### Files That Import `validation_manager`:
```python
# Old import (to be updated):
from tradingbot.core.validation_manager import ValidationManager

# New import:
from tradingbot.core.validationmanager import ValidationManager
```

**Found in**:
- `tradingbot/ui/app.py:9`

---

## Rename Execution Plan

### Step 1: Verify Current State
```bash
# Check which files exist
ls tradingbot/core/*manager*.py
ls tradingbot/core/*executor*.py
ls tradingbot/core/*controller*.py
```

### Step 2: Remove Duplicate Underscore Files
```bash
# For files with both versions, remove underscore version
rm tradingbot/core/config_manager.py    # Keep configmanager.py
rm tradingbot/core/data_manager.py      # Keep datamanager.py  
rm tradingbot/core/portfolio_manager.py # Keep portfoliomanager.py
rm tradingbot/core/risk_manager.py      # Keep riskmanager.py
rm tradingbot/core/error_handler.py     # Keep errorhandler.py
```

### Step 3: Rename Underscore-Only Files
```bash
# For files that only exist with underscores
mv tradingbot/core/trade_executor.py tradingbot/core/tradeexecutor.py
mv tradingbot/core/validation_manager.py tradingbot/core/validationmanager.py  
mv tradingbot/core/runtime_controller.py tradingbot/core/runtimecontroller.py
```

### Step 4: Update Import Statements
Use the unified diff patches below to update all import statements.

---

## Import Fix Patches

### Patch 1: Update config_manager imports
```diff
--- a/tradingbot/core/data_manager.py
+++ b/tradingbot/core/datamanager.py
@@ -1,7 +1,7 @@
 """DataManager module for handling market data operations."""
 import pandas as pd
 import ccxt
-from tradingbot.core.config_manager import ConfigManager
+from tradingbot.core.configmanager import ConfigManager
 
 class DataManager:
```

### Patch 2: Update validation_manager imports  
```diff
--- a/tradingbot/ui/app.py
+++ b/tradingbot/ui/app.py
@@ -6,7 +6,7 @@ from fastapi import FastAPI, HTTPException
 from fastapi.staticfiles import StaticFiles
 
 from tradingbot.core.runtime_controller import RuntimeController
-from tradingbot.core.validation_manager import ValidationManager
+from tradingbot.core.validationmanager import ValidationManager
```

### Patch 3: Update trade_executor imports
```diff
--- a/tradingbot/core/portfolio_manager.py
+++ b/tradingbot/core/portfoliomanager.py
@@ -2,7 +2,7 @@
 import pandas as pd
 from typing import Dict, Any, Optional
 from tradingbot.core.configmanager import ConfigManager
-from tradingbot.core.trade_executor import TradeExecutor
+from tradingbot.core.tradeexecutor import TradeExecutor
```

---

## Validation After Rename

### 1. Import Validation Script
```python
# validate_imports.py
import importlib
import sys

modules_to_test = [
    'tradingbot.core.configmanager',
    'tradingbot.core.datamanager', 
    'tradingbot.core.portfoliomanager',
    'tradingbot.core.riskmanager',
    'tradingbot.core.tradeexecutor',
    'tradingbot.core.validationmanager',
    'tradingbot.core.runtimecontroller',
    'tradingbot.core.errorhandler'
]

for module in modules_to_test:
    try:
        importlib.import_module(module)
        print(f"✅ {module} imports successfully")
    except ImportError as e:
        print(f"❌ {module} failed to import: {e}")
```

### 2. Test Execution
```bash
# Run import validation
python validate_imports.py

# Run existing tests to ensure functionality
python -m pytest tests/ -v

# Check for any remaining underscore imports
grep -r "_manager\|_executor\|_controller" tradingbot/ --include="*.py"
```

---

## Risk Assessment

### Breaking Changes:
- **ALL import statements** will need updating
- **Development environments** will need updated checkouts
- **Documentation** may reference old module names

### Mitigation:
- **Staged deployment**: Update and test one module at a time
- **Import aliases**: Temporarily support old names during transition
- **Comprehensive testing**: Run full test suite after each rename

### Rollback Plan:
- **Git revert**: All changes are in version control
- **Backup**: Keep copy of original file structure
- **Import restoration**: Revert import statements if issues arise

---

## Timeline

### Immediate (Same Day):
1. Remove duplicate underscore files for modules with both versions
2. Update imports for removed files
3. Test core functionality

### Short Term (1-2 Days):
1. Rename underscore-only files
2. Update all remaining imports
3. Full regression testing
4. Update documentation

### Validation (1 Week):
1. Monitor for any missed import references
2. Update any external documentation
3. Verify CI/CD pipelines work correctly

---

**End of Renames Map**
**Generated**: 2025-08-22
**Breaking Changes**: Yes - All imports require updates
**Estimated Effort**: 4-6 hours for complete migration
**Risk Level**: Medium - Systematic approach with comprehensive testing required