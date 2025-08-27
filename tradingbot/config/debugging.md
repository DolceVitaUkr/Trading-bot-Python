# DEBUGGING.md

## Philosophy & Scope

### Core Principles
- **ONE bug at a time** - Never mix bug fixes with features or refactoring
- **No refactors in bugfix PRs** - Behavioral changes are separate from structural changes  
- **Legacy preservation** - Existing logic stays intact; new logic goes behind feature flags
- **Opt-in changes** - All behavior modifications must be reversible and configurable
- **Minimal surface area** - Touch ≤2 files, ≤40 lines per bug fix
- **Evidence-based debugging** - Every hypothesis must be testable and measurable

### Change Categories
- **Bug Fix**: Restore expected behavior without changing interfaces
- **New Logic**: Add capabilities behind feature flags (`FEATURE_NEW_*`)
- **Refactor**: Structure changes with identical behavior (separate PR)

### Debugging Mindset
- **Assume nothing** - Verify every assumption with logs or tests
- **Isolate variables** - Change one thing at a time
- **Preserve crime scene** - Don't modify failing state until captured
- **Follow the data** - Let metrics and logs guide investigation, not intuition

---

## Bug Report Template

Copy-paste this template for all bug reports:

```markdown
## Bug Report

**Summary:** [One sentence description]

**Expected Behavior:**
[What should happen - be specific with examples]

**Actual Behavior:**
[What actually happens - include error messages verbatim]

**Steps to Reproduce:**
1. [Exact commands or UI actions]
2. [Input data or configuration]
3. [Trigger action]
4. [Observe failure]

**Full Stack Trace:**
```
[Complete error traceback - do not truncate]
```

**Environment:**
- Python version: `python --version`
- OS: [Linux/Windows/macOS + version]
- Key dependencies: `pip freeze | grep -E "(trading|bybit|ibkr|pandas|numpy|asyncio)"`
- Configuration: [Relevant config file values]

**Recent Changes:**
[Last 3 commits, config modifications, or deployment changes]

**Minimal Reproducer:**
[Link to gist/snippet that fails in <20 lines]

**Impact & Urgency:**
- [ ] Trading halted (P0 - fix immediately)
- [ ] Paper mode broken (P1 - fix today)  
- [ ] UI degraded (P2 - fix this week)
- [ ] Performance regression (P2-P3 depending on severity)
- [ ] Test flakiness (P3 - fix when convenient)

**Debug Artifacts:**
[Links to log files, screenshots, data dumps]
```

---

## Minimal Reproducer Rule

### Creating Failing Scripts
Every bug needs a standalone reproducer that anyone can run:

```python
# reproducer_[bug_id].py
"""
Minimal reproducer for bug #[ID]
Run with: python reproducer_[bug_id].py
Should fail with: [expected error]
"""
import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    # Minimal setup - no external dependencies
    from core.module import FailingClass
    
    # Create minimal failing case
    obj = FailingClass(param="test_value")
    
    # This should work but fails
    result = obj.method_that_fails()
    print(f"Result: {result}")  # Should print X but prints Y

if __name__ == "__main__":
    main()
```

### Data Fixtures Guidelines
- **Deterministic**: Fixed timestamps, no random seeds, predictable ordering
- **Minimal**: ≤50 rows for CSV fixtures, ≤10 KB files
- **Isolated**: No external API calls, no file system dependencies
- **Versioned**: `fixtures/bug_123/` for each reproducer

### Debug Data Templates
```python
# fixtures/debug_data.py
"""Standard debug datasets for reproducible testing"""

MINIMAL_OHLCV = [
    {"timestamp": 1609459200, "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 1000},
    {"timestamp": 1609459260, "open": 102.0, "high": 108.0, "low": 98.0, "close": 106.0, "volume": 1200},
    {"timestamp": 1609459320, "open": 106.0, "high": 110.0, "low": 103.0, "close": 104.0, "volume": 800},
]

EDGE_CASE_BALANCE = {
    "zero_balance": 0.0,
    "tiny_balance": 0.01, 
    "small_balance": 10.0,
    "negative_balance": -5.0,
}

BROKEN_CONFIG = {
    "missing_api_key": {"api_secret": "test", "api_key": None},
    "invalid_timeframe": {"timeframe": "invalid"},
    "wrong_type": {"max_positions": "not_a_number"},
}
```

---

## Advanced AI Workflow

### Triage Prompt Template
```
CONTEXT: [Brief bug description]
FILES: [List 1-3 most relevant files]
SYMPTOMS: [Observable failures, error messages, unexpected outputs]

Analysis request:
1. Generate 2-3 testable hypotheses ranked by likelihood
2. Suggest one targeted probe to confirm/deny top hypothesis  
3. Provide unified diff ≤40 lines affecting ≤2 files
4. Explain why this approach vs alternatives

CONSTRAINTS:
- NO architectural suggestions
- NO refactoring recommendations  
- Focus on minimal behavioral fixes only
```

### Hypothesis-Driven Debug Prompt
```
BUG: [One sentence summary]
HYPOTHESIS: [Your theory about root cause]

Design experiment to test this hypothesis:
1. What should we observe if hypothesis is correct?
2. What logs/assertions/tests would prove/disprove it?
3. What's the minimal code change to test the fix?
4. How do we ensure no side effects?

Provide specific probe code, not general debugging advice.
```

### Patch Generation Prompt  
```
CONFIRMED ROOT CAUSE: [From hypothesis testing]
FAILING TEST: [Paste the test that reproduces the bug]

Generate patch requirements:
- Unified diff format only
- Preserve all function signatures
- No import reorganization
- No unrelated cleanup
- Add feature flag if behavior changes
- Include reasoning for each changed line

Template: ```diff [your patch] ```
```

### Root Cause Analysis Prompt
```
FIXED BUG: [Description]
APPLIED PATCH: [The diff that fixed it]

Provide post-mortem analysis:
1. **Root cause**: Why did this bug exist? (design flaw, edge case, assumption)
2. **Contributing factors**: What made this bug likely? (complexity, missing tests, unclear specs)
3. **Prevention strategy**: How do we prevent similar bugs? (new test pattern, design principle, code review focus)
4. **Monitoring**: What metrics/alerts would catch this class of bug earlier?

Keep each point to 1-2 sentences max.
```

---

## Coexistence Policy (Enhanced)

### Feature Flag Architecture
```python
# config/feature_flags.py
"""
Centralized feature flag management for safe deployment of new logic.
"""
import os
import logging
from typing import Dict, Set
from dataclasses import dataclass
from enum import Enum

class RolloutStage(Enum):
    DISABLED = "disabled"
    DEV_ONLY = "dev"  
    SHADOW = "shadow"      # Run both, use legacy result
    CANARY = "canary"      # 5% of traffic
    ENABLED = "enabled"

@dataclass
class FeatureFlag:
    name: str
    stage: RolloutStage
    description: str
    rollback_safe: bool = True
    
class FeatureFlags:
    def __init__(self):
        self._flags = self._load_flags()
        self._log_enabled_flags()
    
    def _load_flags(self) -> Dict[str, FeatureFlag]:
        return {
            "NEW_SIGNAL_PROCESSING": FeatureFlag(
                name="NEW_SIGNAL_PROCESSING",
                stage=RolloutStage(os.getenv('FEATURE_NEW_SIGNAL_PROCESSING', 'disabled')),
                description="Enhanced ML signal processing pipeline",
            ),
            "NEW_RISK_MODEL": FeatureFlag(
                name="NEW_RISK_MODEL", 
                stage=RolloutStage(os.getenv('FEATURE_NEW_RISK_MODEL', 'disabled')),
                description="Updated risk calculation with volatility adjustment",
            ),
        }
    
    def is_enabled(self, flag_name: str) -> bool:
        flag = self._flags.get(flag_name)
        if not flag:
            return False
        return flag.stage in [RolloutStage.ENABLED, RolloutStage.CANARY]
    
    def is_shadow_mode(self, flag_name: str) -> bool:
        flag = self._flags.get(flag_name)
        return flag and flag.stage == RolloutStage.SHADOW
    
    def _log_enabled_flags(self):
        enabled = [name for name, flag in self._flags.items() if self.is_enabled(name)]
        if enabled:
            logging.info(f"Feature flags enabled: {enabled}")

# Global instance
feature_flags = FeatureFlags()
```

### Dual-Execution Pattern
```python
# core/signal_processor.py
from config.feature_flags import feature_flags
import logging

class SignalProcessor:
    def process_signals(self, raw_data):
        """Process trading signals with optional new logic"""
        
        if feature_flags.is_shadow_mode("NEW_SIGNAL_PROCESSING"):
            # Run both, compare results, use legacy
            legacy_result = self._process_signals_legacy(raw_data)
            new_result = self._process_signals_v2(raw_data)
            
            self._compare_and_log_results(legacy_result, new_result)
            return legacy_result
            
        elif feature_flags.is_enabled("NEW_SIGNAL_PROCESSING"):
            return self._process_signals_v2(raw_data)
        else:
            return self._process_signals_legacy(raw_data)
    
    def _process_signals_legacy(self, raw_data):
        """Original signal processing - NEVER MODIFY"""
        # Original implementation preserved exactly
        pass
    
    def _process_signals_v2(self, raw_data):
        """New signal processing behind feature flag"""
        # New implementation
        pass
    
    def _compare_and_log_results(self, legacy, new):
        """Log differences for shadow mode analysis"""
        if legacy != new:
            logging.info(f"Signal processing diff: legacy={legacy[:3]}, new={new[:3]}")
```

### Branch-by-Abstraction Pattern
```python
# For complex changes, use abstraction to maintain both paths
class RiskCalculatorInterface:
    def calculate_risk(self, portfolio, market_data): pass

class LegacyRiskCalculator(RiskCalculatorInterface):
    """Original risk calculation - frozen implementation"""
    def calculate_risk(self, portfolio, market_data):
        # Original logic preserved
        pass

class EnhancedRiskCalculator(RiskCalculatorInterface): 
    """New risk calculation with volatility adjustment"""
    def calculate_risk(self, portfolio, market_data):
        # New logic
        pass

def get_risk_calculator():
    if feature_flags.is_enabled("NEW_RISK_MODEL"):
        return EnhancedRiskCalculator()
    return LegacyRiskCalculator()
```

---

## Verification Checklists (Enhanced)

### Pre-Commit Checklist
```bash
# Automated pre-commit verification
#!/bin/bash
set -e

echo "🔍 Pre-commit verification..."

# Code quality
echo "  → Formatting..."
black --check --line-length=100 src/ tests/
ruff check src/ tests/ --fix

echo "  → Type checking..."
mypy src/ --strict --show-error-codes

echo "  → Security scan..."
bandit -r src/ -f json -o security-report.json
safety check --json

echo "  → Quick tests..."
pytest tests/unit/ -x --tb=short -q --durations=5

echo "✅ Pre-commit checks passed"
```

### Debug Session Checklist
Before starting any debugging session:

- [ ] **State preservation**: Save current logs, config, data state
- [ ] **Environment capture**: Document exact versions, settings, feature flags
- [ ] **Hypothesis formation**: Write down 2-3 theories before investigating
- [ ] **Success criteria**: Define what "fixed" looks like specifically
- [ ] **Rollback plan**: Know how to undo any changes quickly
- [ ] **Time box**: Set 2-hour limit before escalating or asking for help

### Before-Merge Checklist
- [ ] **Regression test**: New test fails without patch, passes with patch
- [ ] **Full test suite**: All existing tests remain green  
- [ ] **Coverage verification**: Changed lines have ≥95% coverage
- [ ] **No scope creep**: Only bug-related changes, no "while we're here" fixes
- [ ] **Feature flag validation**: If behavior changes, flag is properly implemented
- [ ] **Documentation updated**: CHANGELOG, README, inline comments updated
- [ ] **Performance check**: No significant performance regression
- [ ] **Backward compatibility**: Old behavior preserved when flag is off

### After-Merge Monitoring
- [ ] **Deployment verification**: Bug fix deployed successfully
- [ ] **Metric stability**: Key trading metrics unchanged for 4 hours
- [ ] **Error rate monitoring**: No new errors introduced
- [ ] **Kill switch tested**: Emergency stop procedures verified
- [ ] **Feature flag toggle**: Verified flag can be turned on/off safely
- [ ] **Rollback rehearsal**: Practice reverting the change

---

## Specialized Debug Playbooks

### Import/Module Resolution Errors
When encountering `ModuleNotFoundError`, `ImportError`, or circular imports:

**Step 1: Environment Diagnosis**
```bash
# Capture complete environment state
python -c "
import sys
print('=== PYTHON PATH ===')
for i, path in enumerate(sys.path):
    print(f'{i}: {path}')

print('\n=== WORKING DIRECTORY ===')
import os
print(f'CWD: {os.getcwd()}')

print('\n=== PROJECT STRUCTURE ===')
for root, dirs, files in os.walk('.'):
    level = root.replace('.', '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files[:5]:  # Limit output
        if file.endswith('.py'):
            print(f'{subindent}{file}')
" > debug_env.txt
```

**Step 2: Import Chain Analysis**
```python
# debug_imports.py
def trace_import_chain(module_name):
    """Trace where import failure occurs in chain"""
    parts = module_name.split('.')
    
    for i in range(1, len(parts) + 1):
        partial_import = '.'.join(parts[:i])
        try:
            __import__(partial_import)
            print(f"✅ {partial_import}")
        except ImportError as e:
            print(f"❌ {partial_import}: {e}")
            break
        except Exception as e:
            print(f"⚠️  {partial_import}: {type(e).__name__}: {e}")
            break

# Usage
trace_import_chain('trading_bot.core.strategy_manager')
```

**Step 3: Circular Import Detection**
```python
# debug_circular.py
import sys
import importlib

def detect_circular_imports(start_module):
    """Find circular import dependencies"""
    importing_stack = []
    
    class ImportTracker:
        def find_module(self, name, path=None):
            if name in importing_stack:
                cycle = ' → '.join(importing_stack + [name])
                print(f"🔄 Circular import detected: {cycle}")
            return None
    
    sys.meta_path.insert(0, ImportTracker())
    
    try:
        importlib.import_module(start_module)
    finally:
        sys.meta_path.remove(sys.meta_path[0])
```

### Dependency Conflict Resolution
When seeing version incompatibilities:

**Step 1: Conflict Analysis**
```bash
# Create dependency graph
pip install pipdeptree
pipdeptree --json > deps.json
pipdeptree --graph-output png > deps.png

# Find conflicts
pip check 2>&1 | tee dependency_conflicts.txt
```

**Step 2: Minimal Version Discovery**
```bash
# Test minimal compatible versions
python -c "
import pkg_resources

def find_minimal_version(package_name, import_test):
    versions = ['1.0.0', '1.5.0', '2.0.0', '2.5.0']  # Adjust range
    for version in versions:
        try:
            pkg_resources.require(f'{package_name}>={version}')
            exec(import_test)
            print(f'✅ {package_name}>={version} works')
            break
        except:
            print(f'❌ {package_name}>={version} fails')

find_minimal_version('pandas', 'import pandas as pd; pd.DataFrame([1,2,3])')
"
```

**Step 3: Clean Environment Testing**
```bash
# Isolated dependency testing
python -m venv debug_clean
source debug_clean/bin/activate
pip install --no-cache-dir [single-package-to-test]
python -c "import [package]; print('Success')"
deactivate
rm -rf debug_clean
```

### Async/Concurrency/Race Condition Debugging
For timing-related bugs and async issues:

**Step 1: Timeline Logging**
```python
# utils/debug_timing.py
import time
import threading
import asyncio
from collections import defaultdict

class TimelineLogger:
    def __init__(self):
        self.events = []
        self.start_time = time.perf_counter()
    
    def log(self, event, context=None):
        elapsed = time.perf_counter() - self.start_time
        thread_id = threading.get_ident()
        
        if asyncio.iscoroutinefunction:
            task_name = getattr(asyncio.current_task(), 'get_name', lambda: 'unknown')()
        else:
            task_name = 'sync'
            
        self.events.append({
            'time': elapsed,
            'event': event,
            'thread': thread_id,
            'task': task_name,
            'context': context
        })
    
    def dump_timeline(self):
        for event in sorted(self.events, key=lambda x: x['time']):
            print(f"[{event['time']:8.4f}] T{event['thread']}/{event['task']}: {event['event']}")

# Global logger for debugging
timeline = TimelineLogger()
```

**Step 2: Deterministic Async Testing**
```python
# tests/utils/deterministic_async.py
import asyncio
from unittest.mock import AsyncMock

class DeterministicClock:
    """Controllable time for async testing"""
    def __init__(self):
        self.current_time = 0
        self.scheduled_callbacks = []
    
    def sleep(self, duration):
        """Mock asyncio.sleep with controllable time"""
        self.current_time += duration
        return AsyncMock(return_value=None)()
    
    def advance_to(self, target_time):
        """Jump to specific time and trigger callbacks"""
        self.current_time = target_time
        # Process any scheduled callbacks
        
def deterministic_test():
    """Template for deterministic async testing"""
    clock = DeterministicClock()
    
    with patch('asyncio.sleep', clock.sleep):
        # Your async test logic here
        pass
```

**Step 3: Race Condition Detection**
```python
# utils/race_detector.py
import threading
import time
from contextlib import contextmanager

class RaceDetector:
    def __init__(self):
        self.shared_state_access = defaultdict(list)
        self.lock = threading.Lock()
    
    @contextmanager
    def monitor_access(self, resource_name, access_type='read'):
        thread_id = threading.get_ident()
        timestamp = time.perf_counter()
        
        with self.lock:
            self.shared_state_access[resource_name].append({
                'thread': thread_id,
                'type': access_type,
                'time': timestamp
            })
        
        yield
        
    def report_potential_races(self):
        """Find overlapping read/write access"""
        for resource, accesses in self.shared_state_access.items():
            writes = [a for a in accesses if a['type'] == 'write']
            reads = [a for a in accesses if a['type'] == 'read']
            
            for write in writes:
                concurrent_reads = [r for r in reads if abs(r['time'] - write['time']) < 0.001]
                if concurrent_reads:
                    print(f"⚠️  Potential race on {resource}: write from thread {write['thread']} concurrent with reads")

# Usage in suspicious code
race_detector = RaceDetector()

def risky_function():
    with race_detector.monitor_access('portfolio_balance', 'write'):
        # Modify shared state
        pass
```

### Performance Regression Analysis
When code suddenly becomes slow:

**Step 1: Automated Performance Profiling**
```python
# utils/perf_monitor.py
import cProfile
import pstats
import io
from functools import wraps

def profile_function(sort_by='tottime', lines_to_show=20):
    """Decorator to profile function performance"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()
            
            try:
                result = func(*args, **kwargs)
            finally:
                pr.disable()
            
            # Capture profile output
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats(sort_by)
            ps.print_stats(lines_to_show)
            
            # Log slow functions
            profile_output = s.getvalue()
            if any(line for line in profile_output.split('\n') if 'function calls' in line and '0.1' in line):
                logging.warning(f"Performance concern in {func.__name__}:\n{profile_output}")
            
            return result
        return wrapper
    return decorator

# Usage
@profile_function(sort_by='cumtime', lines_to_show=10)
def slow_calculation():
    # Your potentially slow code
    pass
```

**Step 2: Performance Regression Testing**
```python
# tests/performance/test_benchmarks.py
import pytest
import time

class TestPerformanceBenchmarks:
    """Performance regression tests - fail if code gets too slow"""
    
    @pytest.mark.performance
    def test_signal_processing_speed(self, sample_market_data):
        """Signal processing should complete within 100ms"""
        processor = SignalProcessor()
        
        start_time = time.perf_counter()
        result = processor.process_signals(sample_market_data)
        elapsed = time.perf_counter() - start_time
        
        # Fail if significantly slower than baseline
        assert elapsed < 0.1, f"Signal processing took {elapsed:.3f}s, expected <0.1s"
        assert result is not None, "Should return valid signals"
    
    @pytest.mark.performance  
    def test_memory_usage_stable(self):
        """Memory usage should not grow unbounded"""
        import psutil
        process = psutil.Process()
        
        initial_memory = process.memory_info().rss
        
        # Run operation multiple times
        for _ in range(100):
            # Your operation here
            pass
        
        final_memory = process.memory_info().rss
        memory_growth = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        assert memory_growth < 50, f"Memory grew by {memory_growth:.1f}MB, expected <50MB"
```

### Data Pipeline Debugging
For data processing and validation issues:

**Step 1: Data State Inspection**
```python
# utils/data_inspector.py
import pandas as pd
import json
from datetime import datetime

def inspect_dataframe(df, name="DataFrame", sample_rows=5):
    """Comprehensive DataFrame debugging output"""
    print(f"\n=== {name.upper()} INSPECTION ===")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    print(f"\nColumns & Types:")
    for col in df.columns:
        dtype = df[col].dtype
        null_count = df[col].isnull().sum() 
        print(f"  {col}: {dtype} (null: {null_count})")
    
    print(f"\nSample data ({sample_rows} rows):")
    print(df.head(sample_rows).to_string())
    
    print(f"\nData quality issues:")
    # Check for common issues
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"  ⚠️  {duplicates} duplicate rows")
    
    for col in df.select_dtypes(include=['number']).columns:
        if (df[col] == 0).all():
            print(f"  ⚠️  {col} is all zeros")
        if df[col].isnull().any():
            print(f"  ⚠️  {col} has {df[col].isnull().sum()} null values")

def save_debug_dataframe(df, context=""):
    """Save DataFrame state for later analysis"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"debug_data_{context}_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"Debug data saved: {filename}")
    return filename
```

**Step 2: Data Schema Validation**
```python
# utils/schema_validator.py
from typing import Dict, Any, List
import pandas as pd

class DataValidator:
    """Validate data conforms to expected schema"""
    
    @staticmethod
    def validate_ohlcv_data(df: pd.DataFrame) -> List[str]:
        """Validate OHLCV market data format"""
        errors = []
        
        # Required columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            errors.append(f"Missing required columns: {missing}")
        
        if not errors:  # Only check data if columns exist
            # Data integrity checks
            if (df['high'] < df['low']).any():
                errors.append("High prices below low prices detected")
            
            if (df['high'] < df['open']).any() or (df['high'] < df['close']).any():
                errors.append("High prices below open/close detected")
            
            if (df['low'] > df['open']).any() or (df['low'] > df['close']).any():
                errors.append("Low prices above open/close detected")
            
            if (df['volume'] < 0).any():
                errors.append("Negative volume detected")
                
        return errors
    
    @staticmethod 
    def validate_signal_data(signals: Dict[str, Any]) -> List[str]:
        """Validate trading signal format"""
        errors = []
        
        required_keys = ['action', 'confidence', 'timestamp']
        missing = [key for key in required_keys if key not in signals]
        if missing:
            errors.append(f"Missing signal keys: {missing}")
        
        if 'confidence' in signals:
            conf = signals['confidence']
            if not isinstance(conf, (int, float)) or not 0 <= conf <= 1:
                errors.append(f"Invalid confidence: {conf} (must be 0-1)")
        
        if 'action' in signals:
            valid_actions = ['buy', 'sell', 'hold']
            if signals['action'] not in valid_actions:
                errors.append(f"Invalid action: {signals['action']} (must be {valid_actions})")
        
        return errors
```

### State Debugging and Persistence
For complex state-related bugs:

**Step 1: State Snapshot Capture**
```python
# utils/state_debugger.py
import pickle
import json
from datetime import datetime
from pathlib import Path

class StateDebugger:
    def __init__(self, debug_dir="debug_states"):
        self.debug_dir = Path(debug_dir)
        self.debug_dir.mkdir(exist_ok=True)
    
    def capture_state(self, obj, context="", method="pickle"):
        """Capture object state for later analysis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"state_{context}_{timestamp}"
        
        if method == "pickle":
            filepath = self.debug_dir / f"{filename}.pkl"
            with open(filepath, 'wb') as f:
                pickle.dump(obj, f)
        elif method == "json":
            filepath = self.debug_dir / f"{filename}.json"
            with open(filepath, 'w') as f:
                json.dump(obj, f, indent=2, default=str)
        
        print(f"State captured: {filepath}")
        return filepath
    
    def load_state(self, filepath):
        """Load previously captured state"""
        if filepath.suffix == '.pkl':
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        elif filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                return json.load(f)
    
    def compare_states(self, state1, state2, context=""):
        """Compare two states and highlight differences"""
        print(f"\n=== STATE COMPARISON: {context} ===")
        
        if hasattr(state1, '__dict__') and hasattr(state2, '__dict__'):
            attrs1 = set(state1.__dict__.keys())
            attrs2 = set(state2.__dict__.keys())
            
            # New attributes
            new_attrs = attrs2 - attrs1
            if new_attrs:
                print(f"New attributes: {new_attrs}")
            
            # Removed attributes  
            removed_attrs = attrs1 - attrs2
            if removed_attrs:
                print(f"Removed attributes: {removed_attrs}")
            
            # Changed values
            for attr in attrs1 & attrs2:
                val1 = getattr(state1, attr)
                val2 = getattr(state2, attr)
                if val1 != val2:
                    print(f"Changed {attr}: {val1} → {val2}")

# Usage
debugger = StateDebugger()

# Before operation
debugger.capture_state(trading_engine, "before_signal_processing")

# After operation  
debugger.capture_state(trading_engine, "after_signal_processing")
```

**Step 2: State Transition Monitoring**
```python
# utils/state_monitor.py
from functools import wraps
import copy

def monitor_state_changes(state_attrs):
    """Decorator to monitor state changes in methods"""
    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            # Capture state before
            before_state = {}
            for attr in state_attrs:
                if hasattr(self, attr):
                    before_state[attr] = copy.deepcopy(getattr(self, attr))
            
            # Execute method
            result = method(self, *args, **kwargs)
            
            # Check for changes
            changes = {}
            for attr in state_attrs:
                if hasattr(self, attr):
                    current_value = getattr(self, attr)
                    if attr in before_state and before_state[attr] != current_value:
                        changes[attr] = {
                            'before': before_state[attr],
                            'after': current_value
                        }
            
            if changes:
                print(f"🔄 State changes in {method.__name__}:")
                for attr, change in changes.items():
                    print(f"  {attr}: {change['before']} → {change['after']}")
            
            return result
        return wrapper
    return decorator

# Usage
class TradingEngine:
    @monitor_state_changes(['position_count', 'total_balance', 'active_orders'])
    def process_signal(self, signal):
        # Method implementation
        pass
```

---

## Prompts Library (Enhanced)

### Advanced Triage Prompt
```
DEBUGGING SESSION: [Bug ID or short description]

CONTEXT:
- System: [Trading bot, ML pipeline, API integration, etc.]
- Component: [Specific module/class/function]  
- Symptoms: [Observable failures, performance issues, data corruption]
- Frequency: [Always, intermittent, specific conditions]
- Recent changes: [Last 3 commits or deployments]

ANALYSIS REQUEST:
1. **Hypotheses**: Generate 3 ranked theories (most likely first)
2. **Diagnostic probe**: Suggest ONE targeted test to confirm top hypothesis
3. **Risk assessment**: Rate impact (P0-P3) and blast radius (local/system-wide)
4. **Minimal fix**: Unified diff ≤40 lines, ≤2 files
5. **Validation plan**: How to verify fix works and doesn't break anything else

CONSTRAINTS:
- NO refactoring suggestions  
- NO "while we're here" improvements
- Focus on restoring expected behavior only
- Preserve all existing interfaces
```

### Root Cause Deep-Dive Prompt
```
BUG ANALYSIS COMPLETE: [Brief description]

EVIDENCE GATHERED:
- Failing test: [Test that reproduces bug]
- Stack trace: [Key parts of error trace]
- State inspection: [Relevant variable values]
- Timeline: [Sequence of events leading to failure]

ROOT CAUSE ANALYSIS REQUEST:
1. **Primary cause**: What specific code/logic caused this failure?
2. **Contributing factors**: What conditions made this bug possible?
3. **Failure mode**: Why didn't existing safeguards catch this?
4. **Blast radius**: What other code might have similar vulnerabilities?
5. **Prevention strategy**: What changes prevent this class of bug?

Provide technical details, not generic advice. Reference specific code patterns and architectural decisions.
```

### Performance Bottleneck Prompt
```
PERFORMANCE REGRESSION DETECTED

BASELINE: [Function X took Y ms before change Z]
CURRENT: [Function X now takes Y*N ms after change Z]
PROFILING DATA: [Top 5 slowest functions from cProfile]

OPTIMIZATION REQUEST:
1. **Bottleneck identification**: Which specific operations are slow?
2. **Algorithmic analysis**: Is this O(n) → O(n²) complexity increase?  
3. **Resource analysis**: CPU bound, I/O bound, or memory bound?
4. **Minimal optimization**: ≤20 line patch to restore performance
5. **Regression guard**: Test to prevent future slowdowns

CONSTRAINTS:
- NO algorithm rewrites
- NO data structure changes
- Focus on tactical optimizations only
- Must maintain identical behavior
```

### Data Corruption Investigation Prompt
```
DATA CORRUPTION INCIDENT

SYMPTOMS: [Describe corrupted data - wrong values, missing records, format issues]
AFFECTED SCOPE: [How much data, which time range, which components]
DETECTION: [How was corruption discovered - alerts, user reports, validation failures]

INVESTIGATION REQUEST:
1. **Corruption pattern**: Is data systematically wrong or randomly corrupted?
2. **Entry point**: Where does bad data first enter the system?
3. **Propagation path**: How does corruption spread through pipeline?
4. **Data recovery**: Can we recover from backups or recalculate?
5. **Immediate mitigation**: How to stop further corruption right now?

URGENCY: Data corruption can compound quickly - prioritize stopping the bleeding over perfect analysis.
```

### Concurrency Bug Analysis Prompt  
```
CONCURRENCY ISSUE DETECTED

SYMPTOMS: [Race conditions, deadlocks, inconsistent state, sporadic failures]
ENVIRONMENT: [Threading, asyncio, multiprocessing details]
REPRODUCTION: [How consistently can you reproduce the issue?]

CONCURRENCY ANALYSIS REQUEST:
1. **Shared resources**: What data is accessed by multiple threads/processes?
2. **Synchronization gaps**: Where are critical sections not properly protected?
3. **Ordering dependencies**: What assumptions about execution order exist?
4. **Deterministic reproduction**: How to make this bug happen reliably?
5. **Surgical fix**: Minimal locking/synchronization to add

Focus on shared mutable state and identify the smallest possible critical sections.
```

### Emergency Production Debug Prompt
```
PRODUCTION EMERGENCY - LIVE TRADING IMPACTED

SEVERITY: [P0 - Trading halted | P1 - Degraded performance | P2 - Data issues]
IMPACT: [Monetary impact, affected users, system components down]
TIMELINE: [When did issue start, escalation path, business deadlines]

EMERGENCY RESPONSE REQUEST:
1. **Immediate stop-gap**: Quick fix to restore service (even if not perfect)
2. **Risk assessment**: What could make this worse if we act?
3. **Rollback options**: Can we revert recent changes safely?
4. **Monitoring**: What metrics confirm fix is working?
5. **Communication plan**: What to tell stakeholders about timeline?

PRIORITY: Restore service first, perfect solution second. Document everything for post-mortem.
```

---

## Change Budgets & Governance (Enhanced)

### Pull Request Size Limits
```yaml
# .github/pr-size-limits.yml
rules:
  bug_fix:
    max_files: 3
    max_lines_changed: 150
    max_functions_modified: 5
    allowed_file_types: [".py", ".yml", ".md"]
    
  feature_flag_addition:
    max_files: 5
    max_lines_changed: 300
    requires_tests: true
    requires_rollback_plan: true
    
  emergency_hotfix:
    max_files: 2
    max_lines_changed: 50
    requires_incident_number: true
    bypass_review: false  # Still needs review, just expedited
```

### Legacy Protection Rules
```python
# scripts/legacy_protection.py
"""
Validate that legacy behavior is preserved
"""
import ast
import sys
from pathlib import Path

PROTECTED_MODULES = [
    "core/legacy_signal_processor.py",
    "core/legacy_risk_calculator.py", 
    "core/legacy_position_manager.py"
]

FORBIDDEN_CHANGES = [
    "function signature modification",
    "class inheritance changes", 
    "public method removal",
    "default value changes"
]

def validate_legacy_preservation(changed_files):
    """Ensure legacy modules are not modified"""
    violations = []
    
    for file_path in changed_files:
        if any(protected in file_path for protected in PROTECTED_MODULES):
            violations.append(f"❌ Attempted to modify protected legacy file: {file_path}")
    
    return violations

def validate_feature_flag_usage(diff_content):
    """Ensure new behavior is behind feature flags"""
    # Parse diff to find behavior changes
    # Check if feature flag gates are present
    pass
```

### Breaking Change Workflow
```markdown
# Breaking Change RFC Template

## RFC: [Change Description]

**Type**: [Bug Fix | Feature Addition | Architecture Change]
**Risk Level**: [Low | Medium | High | Critical]
**Rollback Complexity**: [Simple toggle | Code revert | Data migration]

### Motivation
[Why is this change necessary? What problem does it solve?]

### Detailed Design
[How will the change be implemented? Include code examples.]

### Impact Assessment
- **Affected Components**: [List modules/classes/functions]
- **Breaking Changes**: [What existing behavior will change?]
- **Migration Path**: [How do users adapt to the change?]
- **Rollback Plan**: [How to undo if things go wrong?]

### Feature Flag Strategy
```python
# Implementation approach
if FEATURE_NEW_BEHAVIOR:
    return new_implementation()
else:
    return legacy_implementation()  # Preserved unchanged
```

### Testing Strategy
- [ ] Legacy behavior tests (must continue passing)
- [ ] New behavior tests  
- [ ] Integration tests for both modes
- [ ] Performance comparison tests
- [ ] Rollback scenario tests

### Monitoring & Metrics
- [What metrics will track success/failure?]
- [What alerts will detect issues?]
- [What dashboards will show impact?]

### Timeline
- **Development**: [Duration for implementation]
- **Shadow Mode**: [Duration running both implementations]  
- **Canary Rollout**: [Gradual percentage rollout plan]
- **Full Deployment**: [When to make new behavior default]
```

---

## Templates (Enhanced)

### Comprehensive Bug Report Template
```markdown
## 🐛 Bug Report: [Concise One-Line Summary]

### 📊 Impact Assessment  
- **Severity**: [P0-Critical | P1-High | P2-Medium | P3-Low]
- **Component**: [Signal Processing | Risk Management | Order Execution | UI | Data Pipeline]
- **Business Impact**: [Trading halted | Reduced performance | UI issue | Test flakiness]
- **Affected Users**: [All users | Specific subset | Dev team only]

### 🎯 Expected vs Actual Behavior
**Expected**: [Specific description with examples]
**Actual**: [Specific description with examples]  
**Delta**: [Key differences, quantify when possible]

### 🔬 Reproduction Steps
1. **Environment Setup**: [Config, feature flags, data state]
2. **Trigger Action**: [Exact commands, UI clicks, API calls]
3. **Observable Failure**: [Error messages, wrong outputs, performance issues]

**Reproduction Rate**: [Always | 50% of time | Specific conditions only]

### 📋 Technical Details
**Stack Trace**:
```
[Complete error traceback - do not truncate]
```

**Environment**:
- Python: `python --version`
- OS: [Linux distro/Windows/macOS version]  
- Dependencies: `pip freeze | grep -E "(critical|packages)"`
- Feature Flags: [Currently enabled flags]
- Configuration: [Relevant config values]

### 🕐 Timeline & Context
- **First Observed**: [When did this start happening?]
- **Recent Changes**: [Last 3 commits, deployments, config changes]
- **Related Issues**: [Links to similar bugs or incidents]

### 🔧 Debug Artifacts
- **Minimal Reproducer**: [Link to standalone script that demonstrates bug]
- **Log Files**: [Links to relevant logs]
- **Screenshots**: [For UI bugs]
- **Data Samples**: [Small representative datasets]
- **Performance Profiles**: [For performance bugs]

### 💡 Initial Investigation  
- **Hypotheses**: [Your theories about root cause]
- **Attempted Fixes**: [What you've already tried]
- **Workarounds**: [Temporary solutions in use]

### ⚡ Urgency Factors
- [ ] Active trading disrupted
- [ ] Data integrity compromised  
- [ ] Security implications
- [ ] Blocking other development
- [ ] Customer-facing issue
```

### Advanced PR Checklist
```markdown
## 🔍 Pull Request Checklist

### 📝 Change Classification
- [ ] **Bug Fix**: Restores expected behavior without interface changes
- [ ] **New Feature**: Adds capabilities behind feature flags
- [ ] **Refactor**: Structure changes with identical behavior
- [ ] **Performance**: Optimization with behavior preservation
- [ ] **Documentation**: Non-code changes only

### 🛡️ Pre-Merge Verification

#### Code Quality
- [ ] Passes `make lint` (black, ruff, isort)
- [ ] Passes `make typecheck` (mypy --strict)
- [ ] Passes `make test-fast` (unit tests)
- [ ] Passes `make security` (bandit, safety)
- [ ] No merge conflicts with main branch

#### Test Coverage
- [ ] New test fails without patch, passes with patch
- [ ] All existing tests remain green
- [ ] Coverage ≥95% on changed lines
- [ ] Integration tests updated if needed
- [ ] Performance benchmarks unchanged

#### Change Scope
- [ ] Touches ≤3 files (bug fixes: ≤2 files)
- [ ] Changes ≤150 lines total (bug fixes: ≤40 lines)
- [ ] One logical change only
- [ ] No unrelated formatting/cleanup changes
- [ ] No "while we're here" improvements

### 🚩 Feature Flag Requirements (if behavior changes)
- [ ] Feature flag defined in `config/feature_flags.py`  
- [ ] Default value preserves legacy behavior
- [ ] Both modes tested with parametrized tests
- [ ] Documentation explains flag usage
- [ ] Rollback plan documented in PR description

### 📚 Documentation & Communication  
- [ ] CHANGELOG.md updated
- [ ] Inline code comments added for complex logic
- [ ] README updated if public interfaces changed
- [ ] Team notified if breaking changes (even behind flags)

### 🔍 Legacy Preservation (critical for bug fixes)
- [ ] No modifications to files in `PROTECTED_MODULES`
- [ ] No function signature changes
- [ ] No default parameter value changes  
- [ ] Existing behavior preserved when feature flags disabled

### 🎯 Deployment Readiness
- [ ] Rollback plan tested and documented
- [ ] Monitoring alerts configured for new failure modes
- [ ] Performance impact assessed
- [ ] Database migrations (if any) are reversible
```

### Feature Flag Implementation Template
```python
# config/feature_flags.py
"""
Feature flag: FEATURE_NEW_[COMPONENT]_[CAPABILITY]
Description: [What this flag enables]
Default: False (preserves legacy behavior)
Rollout plan: Dev → Shadow → Canary → Full
"""

from enum import Enum
import os
import logging

class RolloutStage(Enum):
    DISABLED = "disabled"
    DEV_ONLY = "dev"
    SHADOW = "shadow"      # Run both paths, use legacy result
    CANARY = "canary"      # Small percentage of traffic
    ENABLED = "enabled"    # Full rollout

class FeatureFlag:
    def __init__(self, name: str, description: str, default_stage: RolloutStage = RolloutStage.DISABLED):
        self.name = name
        self.description = description
        self.stage = RolloutStage(os.getenv(name, default_stage.value))
        
    def is_enabled(self) -> bool:
        return self.stage in [RolloutStage.ENABLED, RolloutStage.CANARY]
    
    def is_shadow_mode(self) -> bool:
        return self.stage == RolloutStage.SHADOW

# Define your feature flags
FEATURE_NEW_SIGNAL_PROCESSING = FeatureFlag(
    name="FEATURE_NEW_SIGNAL_PROCESSING",
    description="Enhanced ML signal processing with volatility adjustment"
)

FEATURE_NEW_RISK_CALCULATION = FeatureFlag(
    name="FEATURE_NEW_RISK_CALCULATION", 
    description="Updated risk model with correlation analysis"
)

# Usage in code
def process_trading_signals(market_data):
    """Process signals with optional enhanced logic"""
    
    if FEATURE_NEW_SIGNAL_PROCESSING.is_shadow_mode():
        # Run both implementations, compare results
        legacy_signals = _process_signals_legacy(market_data)
        new_signals = _process_signals_v2(market_data)
        _log_signal_differences(legacy_signals, new_signals)
        return legacy_signals  # Use legacy result in shadow mode
        
    elif FEATURE_NEW_SIGNAL_PROCESSING.is_enabled():
        return _process_signals_v2(market_data)
    else:
        return _process_signals_legacy(market_data)

def _process_signals_legacy(market_data):
    """Original signal processing - NEVER MODIFY THIS FUNCTION"""
    # Preserved legacy implementation
    pass

def _process_signals_v2(market_data):
    """Enhanced signal processing behind feature flag"""  
    # New implementation
    pass

def _log_signal_differences(legacy, new):
    """Log differences for shadow mode analysis"""
    if legacy != new:
        logging.info(f"Signal processing diff detected: legacy_count={len(legacy)}, new_count={len(new)}")
```

### Guardrail Test Templates
```python
# tests/regression/test_bug_[ID]_regression.py
"""
Regression test for Bug #[ID]: [One-line description]

This test ensures the specific bug cannot reoccur and serves as
documentation of the expected behavior.
"""
import pytest
from unittest.mock import patch, MagicMock

class TestBug[ID]Regression:
    """Regression tests for bug #[ID]"""
    
    def test_bug_[ID]_core_issue(self):
        """
        Test the core issue that caused bug #[ID]
        
        Bug: [Brief description]
        Root cause: [Technical explanation]
        """
        # Setup - minimal reproduction case
        [setup code]
        
        # Execute - the operation that was failing
        result = [function_under_test]([minimal_inputs])
        
        # Verify - check the fix works  
        assert result == [expected_value], f"Bug #[ID] regression: got {result}, expected {expected_value}"
        
        # Additional invariants that should hold
        assert [other_checks_that_should_pass]
    
    def test_bug_[ID]_edge_cases(self):
        """Test edge cases related to bug #[ID]"""
        # Test boundary conditions that might trigger similar issues
        edge_cases = [
            [edge_case_1],
            [edge_case_2], 
            [edge_case_3]
        ]
        
        for case in edge_cases:
            result = [function_under_test](case)
            assert [appropriate_assertion], f"Edge case failed: {case}"
    
    @pytest.mark.parametrize("feature_flag", [False, True])
    def test_bug_[ID]_both_feature_modes(self, monkeypatch, feature_flag):
        """If bug fix involved feature flags, test both modes"""
        monkeypatch.setenv('FEATURE_[NAME]', str(feature_flag).lower())
        
        # Import after setting environment variable
        from [module] import [function]
        
        result = [function]([test_inputs])
        
        # Both modes should work correctly (may have different behavior)
        if feature_flag:
            assert [new_behavior_assertion]
        else:
            assert [legacy_behavior_assertion]
```

### Emergency Hotfix Template
```markdown
# 🚨 EMERGENCY HOTFIX: [Issue Summary]

## Incident Details
- **Incident ID**: [Link to incident management system]
- **Severity**: P0 - Trading Disrupted
- **Impact**: [Specific business impact]
- **Started**: [Timestamp when issue began]
- **Detected**: [Timestamp when issue was discovered]

## Root Cause (Brief)
[One paragraph explaining what broke and why]

## Immediate Fix
This hotfix implements the minimal change to restore service:

```diff
[Show the exact diff - should be very small]
```

## Risk Assessment
- **Change Scope**: [X lines in Y files]
- **Blast Radius**: [What could break if this fix fails]
- **Rollback Plan**: [How to undo this change quickly]
- **Testing**: [What testing was possible in emergency timeframe]

## Post-Deployment Verification
- [ ] Trading resumed successfully
- [ ] Error rates returned to normal
- [ ] Key metrics stable for 30 minutes
- [ ] No new alerts triggered

## Follow-Up Actions
- [ ] Schedule post-mortem meeting
- [ ] Create comprehensive fix to replace hotfix
- [ ] Review why monitoring didn't catch this earlier
- [ ] Update runbooks based on lessons learned

## Approval
- **Incident Commander**: [Name] ✅
- **Technical Lead**: [Name] ✅ 
- **On-Call Engineer**: [Name] ✅

*This is an emergency hotfix. A proper fix with full testing will follow in [timeframe].*
```

---

## Emergency Response Playbook

### P0 - Trading Halted
```bash
#!/bin/bash
# emergency_response.sh

echo "🚨 P0 EMERGENCY RESPONSE ACTIVATED"

# 1. Immediate assessment
echo "Step 1: Assess scope of impact"
echo "  → Check trading engine status"
curl -f http://localhost:8000/health || echo "❌ Trading engine down"
echo "  → Check market data feeds" 
tail -n 10 logs/market_data.log | grep ERROR

# 2. Enable kill switch if not already active
echo "Step 2: Activate kill switch"
export TRADING_KILL_SWITCH=true
echo "TRADING_KILL_SWITCH=true" >> .env

# 3. Preserve current state
echo "Step 3: Preserve debugging state"
mkdir -p emergency_debug/$(date +%Y%m%d_%H%M%S)
cp logs/*.log emergency_debug/$(date +%Y%m%d_%H%M%S)/
cp config/*.yml emergency_debug/$(date +%Y%m%d_%H%M%S)/

# 4. Quick rollback attempt
echo "Step 4: Attempt quick rollback"
read -p "Rollback to last known good version? (y/n): " -n 1 -r
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git log --oneline -5
    read -p "Enter commit hash to rollback to: " commit_hash
    git revert $commit_hash --no-edit
fi

# 5. Communication
echo "Step 5: Notify stakeholders"
echo "  → Update status page"
echo "  → Notify on-call team"
echo "  → Document in incident log"

echo "✅ Emergency response complete. Begin detailed debugging."
```

### Debug Session Checklist
```markdown
## 🔍 Debug Session Prep

### Before Starting (5 minutes)
- [ ] **Environment snapshot**: Git status, feature flags, config values
- [ ] **State preservation**: Backup logs, database state, active positions  
- [ ] **Hypothesis formation**: Write down 2-3 theories before investigating
- [ ] **Success criteria**: Define what "fixed" looks like specifically
- [ ] **Time box**: Set 2-hour limit, escalate if not resolved

### During Debugging
- [ ] **One change at a time**: Test single variables in isolation
- [ ] **Document findings**: Log every test, assumption, and result
- [ ] **Evidence over intuition**: Let data guide decisions, not hunches
- [ ] **Preserve failure state**: Don't "fix" things until you understand them

### After Resolution  
- [ ] **Root cause documented**: Technical explanation of why bug existed
- [ ] **Fix validated**: Comprehensive testing of solution
- [ ] **Regression test added**: Permanent guard against similar issues
- [ ] **Knowledge shared**: Update team on findings and prevention
```

---

*This debugging guide is a living document. Update it as new patterns emerge and lessons are learned from production incidents. The goal is not just to fix bugs, but to build a culture of disciplined, evidence-based debugging that preserves system stability while enabling continuous improvement.*