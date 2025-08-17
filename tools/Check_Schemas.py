#!/usr/bin/env python3
"""
Check_Schemas.py

Validates data schemas across the trading bot for consistency.
Checks:
1. OHLCV data schemas (timestamp, OHLC, volume columns and types)
2. Timezone consistency across datasets
3. Options greeks schema compliance
4. Configuration schema validation
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import sys
import os
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from Data_Registry import Data_Registry


class SchemaChecker:
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.data_registry = Data_Registry
        self.violations = []
        self.datasets_checked = 0
        
        # Define expected schemas
        self.ohlcv_schema = {
            'required_columns': ['timestamp', 'open', 'high', 'low', 'close', 'volume'],
            'numeric_columns': ['open', 'high', 'low', 'close', 'volume'],
            'timestamp_format': 'unix_ms_or_datetime',
            'required_order': True  # OHLC order matters for validation
        }
        
        self.options_schema = {
            'required_columns': ['timestamp', 'strike', 'expiry', 'option_type', 
                               'bid', 'ask', 'delta', 'gamma', 'theta', 'vega', 'rho'],
            'numeric_columns': ['strike', 'bid', 'ask', 'delta', 'gamma', 'theta', 'vega', 'rho'],
            'categorical_columns': {'option_type': ['call', 'put', 'Call', 'Put', 'CALL', 'PUT']},
            'timestamp_format': 'unix_ms_or_datetime'
        }
        
        self.forex_schema = {
            'required_columns': ['timestamp', 'open', 'high', 'low', 'close', 'volume'],
            'numeric_columns': ['open', 'high', 'low', 'close', 'volume'],
            'timestamp_format': 'unix_ms_or_datetime',
            'precision_requirements': {
                'major_pairs': 5,  # EUR/USD, GBP/USD etc.
                'minor_pairs': 4,  # EUR/GBP, AUD/JPY etc.
                'exotic_pairs': 4
            }
        }
    
    def check_all_schemas(self):
        """Main entry point for schema validation"""
        print("🔍 Starting schema validation...")
        
        # Check OHLCV data files
        self._check_ohlcv_data()
        
        # Check options data if present
        self._check_options_data()
        
        # Check configuration schemas
        self._check_config_schemas()
        
        # Check state file schemas
        self._check_state_schemas()
        
        # Generate report
        self._generate_schema_report()
        
        return self.violations
    
    def _check_ohlcv_data(self):
        """Check OHLCV data files for schema compliance"""
        print("📊 Checking OHLCV data schemas...")
        
        # Look for historical data files
        historical_root = self.data_registry.data_root / "historical"
        if not historical_root.exists():
            print("⚠️  No historical data directory found")
            return
        
        csv_files = list(historical_root.rglob("*.csv"))
        print(f"📁 Found {len(csv_files)} CSV files to check")
        
        for csv_file in csv_files:
            try:
                self._validate_ohlcv_file(csv_file)
            except Exception as e:
                self.violations.append({
                    'type': 'file_read_error',
                    'file': str(csv_file.relative_to(self.root_path)),
                    'error': str(e),
                    'severity': 'error'
                })
    
    def _validate_ohlcv_file(self, csv_file: Path):
        """Validate a single OHLCV CSV file"""
        try:
            # Try to read the file
            df = pd.read_csv(csv_file)
            self.datasets_checked += 1
            
            if df.empty:
                self.violations.append({
                    'type': 'empty_dataset',
                    'file': str(csv_file.relative_to(self.root_path)),
                    'description': "Dataset is empty",
                    'severity': 'warning'
                })
                return
            
            file_path = str(csv_file.relative_to(self.root_path))
            
            # Check required columns
            missing_columns = []
            for col in self.ohlcv_schema['required_columns']:
                if col not in df.columns:
                    missing_columns.append(col)
            
            if missing_columns:
                self.violations.append({
                    'type': 'missing_columns',
                    'file': file_path,
                    'missing_columns': missing_columns,
                    'actual_columns': list(df.columns),
                    'severity': 'error'
                })
                return  # Can't continue validation without required columns
            
            # Check data types
            type_issues = self._check_column_types(df, file_path)
            self.violations.extend(type_issues)
            
            # Check timestamp format and timezone
            timestamp_issues = self._check_timestamp_format(df, file_path)
            self.violations.extend(timestamp_issues)
            
            # Check OHLC logical consistency
            ohlc_issues = self._check_ohlc_consistency(df, file_path)
            self.violations.extend(ohlc_issues)
            
            # Check for missing data
            missing_data_issues = self._check_missing_data(df, file_path)
            self.violations.extend(missing_data_issues)
            
        except Exception as e:
            self.violations.append({
                'type': 'validation_error',
                'file': str(csv_file.relative_to(self.root_path)),
                'error': str(e),
                'severity': 'error'
            })
    
    def _check_column_types(self, df: pd.DataFrame, file_path: str) -> List[Dict]:
        """Check that numeric columns are actually numeric"""
        issues = []
        
        for col in self.ohlcv_schema['numeric_columns']:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    # Try to convert and see if there are non-numeric values
                    try:
                        pd.to_numeric(df[col], errors='raise')
                    except (ValueError, TypeError):
                        issues.append({
                            'type': 'invalid_data_type',
                            'file': file_path,
                            'column': col,
                            'expected': 'numeric',
                            'actual': str(df[col].dtype),
                            'sample_values': list(df[col].head().astype(str)),
                            'severity': 'error'
                        })
        
        return issues
    
    def _check_timestamp_format(self, df: pd.DataFrame, file_path: str) -> List[Dict]:
        """Check timestamp format and timezone consistency"""
        issues = []
        
        if 'timestamp' not in df.columns:
            return issues
        
        timestamp_col = df['timestamp']
        
        # Check if timestamps are in a recognizable format
        try:
            # Try to parse as datetime
            if pd.api.types.is_numeric_dtype(timestamp_col):
                # Assume Unix timestamp (check if reasonable range)
                sample_ts = timestamp_col.iloc[0]
                if sample_ts > 1e12:  # Likely milliseconds
                    parsed_dt = pd.to_datetime(timestamp_col, unit='ms', utc=True)
                elif sample_ts > 1e9:  # Likely seconds
                    parsed_dt = pd.to_datetime(timestamp_col, unit='s', utc=True)
                else:
                    issues.append({
                        'type': 'invalid_timestamp_range',
                        'file': file_path,
                        'description': f"Timestamp values seem out of reasonable range: {sample_ts}",
                        'severity': 'warning'
                    })
                    return issues
            else:
                # Try to parse as string
                parsed_dt = pd.to_datetime(timestamp_col, utc=True)
            
            # Check for timezone awareness
            if parsed_dt.dt.tz is None:
                issues.append({
                    'type': 'missing_timezone',
                    'file': file_path,
                    'description': "Timestamps are not timezone-aware",
                    'severity': 'warning'
                })
            
            # Check for reasonable date range
            min_date = parsed_dt.min()
            max_date = parsed_dt.max()
            
            if min_date.year < 2010:
                issues.append({
                    'type': 'suspicious_timestamp',
                    'file': file_path,
                    'description': f"Earliest timestamp is {min_date}, which seems too old for trading data",
                    'severity': 'warning'
                })
            
            if max_date > pd.Timestamp.now(tz='UTC'):
                issues.append({
                    'type': 'future_timestamp',
                    'file': file_path,
                    'description': f"Latest timestamp {max_date} is in the future",
                    'severity': 'error'
                })
            
        except Exception as e:
            issues.append({
                'type': 'timestamp_parse_error',
                'file': file_path,
                'error': str(e),
                'severity': 'error'
            })
        
        return issues
    
    def _check_ohlc_consistency(self, df: pd.DataFrame, file_path: str) -> List[Dict]:
        """Check OHLC logical consistency (High >= Open,Close; Low <= Open,Close)"""
        issues = []
        
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return issues
        
        try:
            # Convert to numeric if not already
            for col in required_cols:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Check high >= max(open, close)
            high_violations = (df['high'] < df[['open', 'close']].max(axis=1)).sum()
            if high_violations > 0:
                issues.append({
                    'type': 'ohlc_logical_error',
                    'file': file_path,
                    'description': f"High price is less than Open or Close in {high_violations} rows",
                    'violation_count': high_violations,
                    'severity': 'error'
                })
            
            # Check low <= min(open, close)
            low_violations = (df['low'] > df[['open', 'close']].min(axis=1)).sum()
            if low_violations > 0:
                issues.append({
                    'type': 'ohlc_logical_error',
                    'file': file_path,
                    'description': f"Low price is greater than Open or Close in {low_violations} rows",
                    'violation_count': low_violations,
                    'severity': 'error'
                })
            
            # Check for negative prices
            negative_prices = (df[required_cols] < 0).any(axis=1).sum()
            if negative_prices > 0:
                issues.append({
                    'type': 'negative_prices',
                    'file': file_path,
                    'description': f"Negative prices found in {negative_prices} rows",
                    'violation_count': negative_prices,
                    'severity': 'error'
                })
            
        except Exception as e:
            issues.append({
                'type': 'ohlc_validation_error',
                'file': file_path,
                'error': str(e),
                'severity': 'error'
            })
        
        return issues
    
    def _check_missing_data(self, df: pd.DataFrame, file_path: str) -> List[Dict]:
        """Check for missing data patterns"""
        issues = []
        
        # Check for NaN values
        null_counts = df.isnull().sum()
        if null_counts.any():
            null_columns = null_counts[null_counts > 0]
            issues.append({
                'type': 'missing_data',
                'file': file_path,
                'null_counts': dict(null_columns),
                'total_rows': len(df),
                'severity': 'warning'
            })
        
        # Check for duplicate timestamps
        if 'timestamp' in df.columns:
            duplicate_count = df['timestamp'].duplicated().sum()
            if duplicate_count > 0:
                issues.append({
                    'type': 'duplicate_timestamps',
                    'file': file_path,
                    'duplicate_count': duplicate_count,
                    'severity': 'warning'
                })
        
        return issues
    
    def _check_options_data(self):
        """Check options data schemas if present"""
        print("📈 Checking options data schemas...")
        
        # Look for options-specific data
        options_path = self.data_registry.data_root / "options"
        if options_path.exists():
            csv_files = list(options_path.rglob("*.csv"))
            for csv_file in csv_files:
                try:
                    self._validate_options_file(csv_file)
                except Exception as e:
                    self.violations.append({
                        'type': 'options_validation_error',
                        'file': str(csv_file.relative_to(self.root_path)),
                        'error': str(e),
                        'severity': 'error'
                    })
    
    def _validate_options_file(self, csv_file: Path):
        """Validate options data file"""
        df = pd.read_csv(csv_file)
        file_path = str(csv_file.relative_to(self.root_path))
        
        # Check required columns for options
        missing_columns = []
        for col in self.options_schema['required_columns']:
            if col not in df.columns:
                missing_columns.append(col)
        
        if missing_columns:
            self.violations.append({
                'type': 'missing_options_columns',
                'file': file_path,
                'missing_columns': missing_columns,
                'severity': 'error'
            })
        
        # Check option type values
        if 'option_type' in df.columns:
            valid_types = self.options_schema['categorical_columns']['option_type']
            invalid_types = df[~df['option_type'].isin(valid_types)]['option_type'].unique()
            if len(invalid_types) > 0:
                self.violations.append({
                    'type': 'invalid_option_types',
                    'file': file_path,
                    'invalid_values': list(invalid_types),
                    'valid_values': valid_types,
                    'severity': 'error'
                })
    
    def _check_config_schemas(self):
        """Check configuration file schemas"""
        print("⚙️  Checking configuration schemas...")
        
        # Check main config.py
        try:
            import config
            self._validate_config_module(config)
        except Exception as e:
            self.violations.append({
                'type': 'config_import_error',
                'file': 'config.py',
                'error': str(e),
                'severity': 'error'
            })
    
    def _validate_config_module(self, config_module):
        """Validate the config module has required settings"""
        required_settings = [
            'PRODUCTS_ENABLED',
            'IBKR_API_MODE', 
            'TRAINING_MODE',
            'LOG_LEVEL'
        ]
        
        missing_settings = []
        for setting in required_settings:
            if not hasattr(config_module, setting):
                missing_settings.append(setting)
        
        if missing_settings:
            self.violations.append({
                'type': 'missing_config_settings',
                'file': 'config.py',
                'missing_settings': missing_settings,
                'severity': 'warning'
            })
    
    def _check_state_schemas(self):
        """Check state file schemas"""
        print("💾 Checking state file schemas...")
        
        # Check for state files
        if self.data_registry.state_root.exists():
            jsonl_files = list(self.data_registry.state_root.rglob("*.jsonl"))
            json_files = list(self.data_registry.state_root.rglob("*.json"))
            
            for file_path in jsonl_files + json_files:
                try:
                    self._validate_state_file(file_path)
                except Exception as e:
                    self.violations.append({
                        'type': 'state_file_error',
                        'file': str(file_path.relative_to(self.root_path)),
                        'error': str(e),
                        'severity': 'warning'
                    })
    
    def _validate_state_file(self, file_path: Path):
        """Validate a state file (JSON/JSONL)"""
        if file_path.suffix == '.jsonl':
            # JSONL file - each line should be valid JSON
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            json.loads(line)
                        except json.JSONDecodeError as e:
                            self.violations.append({
                                'type': 'invalid_jsonl_line',
                                'file': str(file_path.relative_to(self.root_path)),
                                'line': line_num,
                                'error': str(e),
                                'severity': 'error'
                            })
        else:
            # Regular JSON file
            with open(file_path, 'r') as f:
                try:
                    json.load(f)
                except json.JSONDecodeError as e:
                    self.violations.append({
                        'type': 'invalid_json',
                        'file': str(file_path.relative_to(self.root_path)),
                        'error': str(e),
                        'severity': 'error'
                    })
    
    def _generate_schema_report(self):
        """Generate schema validation report"""
        output_path = self.root_path / "artifacts" / "schema_report.md"
        
        with open(output_path, 'w') as f:
            f.write("# Schema Validation Report\n\n")
            f.write(f"Generated: {self._get_timestamp()}\n\n")
            
            # Summary
            f.write("## Summary\n\n")
            f.write(f"- Datasets checked: {self.datasets_checked}\n")
            f.write(f"- Total violations: {len(self.violations)}\n")
            
            # Count by severity
            by_severity = {}
            for violation in self.violations:
                severity = violation.get('severity', 'unknown')
                by_severity[severity] = by_severity.get(severity, 0) + 1
            
            f.write("\n**By severity:**\n")
            for severity, count in sorted(by_severity.items()):
                icon = "❌" if severity == "error" else "⚠️" if severity == "warning" else "ℹ️"
                f.write(f"- {icon} {severity}: {count}\n")
            
            # Count by type
            by_type = {}
            for violation in self.violations:
                vtype = violation['type']
                by_type[vtype] = by_type.get(vtype, 0) + 1
            
            f.write("\n**By type:**\n")
            for vtype, count in sorted(by_type.items()):
                f.write(f"- {vtype}: {count}\n")
            
            # Detailed violations
            if self.violations:
                f.write("\n## Detailed Violations\n\n")
                
                # Group by type
                violations_by_type = {}
                for violation in self.violations:
                    vtype = violation['type']
                    if vtype not in violations_by_type:
                        violations_by_type[vtype] = []
                    violations_by_type[vtype].append(violation)
                
                for vtype, violations in sorted(violations_by_type.items()):
                    f.write(f"### {vtype.replace('_', ' ').title()}\n\n")
                    
                    for violation in violations:
                        severity_icon = "❌" if violation.get('severity') == "error" else "⚠️"
                        f.write(f"{severity_icon} **{violation.get('file', 'Unknown file')}**\n")
                        
                        if 'description' in violation:
                            f.write(f"- {violation['description']}\n")
                        if 'error' in violation:
                            f.write(f"- Error: {violation['error']}\n")
                        if 'missing_columns' in violation:
                            f.write(f"- Missing columns: {violation['missing_columns']}\n")
                        if 'violation_count' in violation:
                            f.write(f"- Affected rows: {violation['violation_count']}\n")
                        
                        f.write("\n")
            else:
                f.write("\n## ✅ No Schema Violations Found\n\n")
                f.write("All checked datasets conform to expected schemas!\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            if by_severity.get('error', 0) > 0:
                f.write("🔴 **Critical Issues:** Fix all errors before using data in production.\n\n")
            if by_severity.get('warning', 0) > 0:
                f.write("🟡 **Warnings:** Review and address warnings for data quality.\n\n")
            
            f.write("### Schema Auto-Fix Suggestions\n\n")
            f.write("1. **Missing columns:** Add default values or regenerate data\n")
            f.write("2. **Data type issues:** Convert strings to numeric where appropriate\n")
            f.write("3. **Timestamp issues:** Standardize on UTC timezone and consistent format\n")
            f.write("4. **OHLC violations:** Review data source and cleaning logic\n")
            f.write("5. **Missing data:** Implement interpolation or forward-fill strategies\n")
        
        print(f"📋 Generated {output_path}")
    
    def _get_timestamp(self):
        """Get current timestamp"""
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def main():
    """Main entry point"""
    root_path = Path(__file__).parent.parent
    checker = SchemaChecker(str(root_path))
    
    violations = checker.check_all_schemas()
    
    # Print summary
    print("\n" + "="*50)
    print("🎯 SCHEMA VALIDATION SUMMARY")
    print("="*50)
    print(f"Datasets checked: {checker.datasets_checked}")
    print(f"Total violations: {len(violations)}")
    
    # Count by severity
    by_severity = {}
    for violation in violations:
        severity = violation.get('severity', 'unknown')
        by_severity[severity] = by_severity.get(severity, 0) + 1
    
    if by_severity:
        print("\nBy severity:")
        for severity, count in sorted(by_severity.items()):
            icon = "❌" if severity == "error" else "⚠️" if severity == "warning" else "ℹ️"
            print(f"  {icon} {severity}: {count}")
    else:
        print("\n✅ All schemas are valid!")
    
    print("\n📁 Detailed report: ./artifacts/schema_report.md")


if __name__ == "__main__":
    main()