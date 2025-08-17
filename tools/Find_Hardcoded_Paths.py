#!/usr/bin/env python3
"""
Find_Hardcoded_Paths.py

Scans the codebase for hardcoded paths and suggests Data_Registry replacements.
Identifies:
- Absolute paths (C:/, /Users/, /home/, /mnt/)
- Relative data paths (backup_data/, data/, logs/, models/)
- Magic strings for file extensions (.csv, .json, .pkl)
- Direct file I/O bypassing data layer
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple
import ast


class HardcodedPathFinder:
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.findings = []
        
        # Patterns to detect hardcoded paths
        self.path_patterns = [
            # Absolute paths
            (r'["\'][C-Z]:\\[^"\']*["\']', "windows_absolute_path"),
            (r'["\']\/[Uu]sers\/[^"\']*["\']', "macos_user_path"),
            (r'["\']\/home\/[^"\']*["\']', "linux_home_path"),
            (r'["\']\/mnt\/[^"\']*["\']', "linux_mount_path"),
            (r'["\']\/tmp\/[^"\']*["\']', "linux_tmp_path"),
            
            # Relative data paths (common problematic patterns)
            (r'["\']backup_data[\/\\][^"\']*["\']', "backup_data_path"),
            (r'["\']data[\/\\][^"\']*["\']', "data_directory_path"),
            (r'["\']logs[\/\\][^"\']*["\']', "logs_directory_path"),
            (r'["\']models[\/\\][^"\']*["\']', "models_directory_path"),
            (r'["\']state[\/\\][^"\']*["\']', "state_directory_path"),
            
            # File extensions as magic strings
            (r'["\'][^"\']*\.csv["\']', "csv_file_extension"),
            (r'["\'][^"\']*\.json["\']', "json_file_extension"),
            (r'["\'][^"\']*\.pkl["\']', "pickle_file_extension"),
            (r'["\'][^"\']*\.log["\']', "log_file_extension"),
            (r'["\'][^"\']*\.jsonl["\']', "jsonl_file_extension"),
        ]
        
        # Direct file I/O functions that should use data layer
        self.file_io_patterns = [
            (r'open\s*\(', "direct_file_open"),
            (r'with\s+open\s*\(', "direct_file_open_context"),
            (r'\.to_csv\s*\(', "pandas_to_csv"),
            (r'\.read_csv\s*\(', "pandas_read_csv"),
            (r'json\.dump[s]?\s*\(', "json_dump"),
            (r'json\.load[s]?\s*\(', "json_load"),
            (r'pickle\.dump\s*\(', "pickle_dump"),
            (r'pickle\.load\s*\(', "pickle_load"),
        ]
    
    def scan(self):
        """Main scanning entry point"""
        print("🔍 Scanning for hardcoded paths and direct file I/O...")
        
        python_files = list(self.root_path.rglob("*.py"))
        print(f"📁 Scanning {len(python_files)} Python files")
        
        for py_file in python_files:
            if self._should_skip_file(py_file):
                continue
                
            try:
                self._scan_file(py_file)
            except Exception as e:
                print(f"⚠️  Error scanning {py_file}: {e}")
        
        # Generate reports
        self._generate_findings_report()
        self._generate_replacement_suggestions()
        
        return self.findings
    
    def _should_skip_file(self, py_file: Path) -> bool:
        """Skip certain files from scanning"""
        skip_patterns = [
            "__pycache__",
            ".venv",
            "venv",
            ".git",
            ".pytest_cache",
            "tools/Find_Hardcoded_Paths.py"  # Skip this file itself
        ]
        
        file_str = str(py_file)
        return any(pattern in file_str for pattern in skip_patterns)
    
    def _scan_file(self, py_file: Path):
        """Scan a single file for hardcoded paths"""
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.splitlines()
            relative_path = str(py_file.relative_to(self.root_path))
            
            # Scan for path patterns
            for line_num, line in enumerate(lines, 1):
                self._scan_line(relative_path, line_num, line)
                
        except Exception as e:
            print(f"❌ Failed to scan {py_file}: {e}")
    
    def _scan_line(self, file_path: str, line_num: int, line: str):
        """Scan a single line for patterns"""
        line_stripped = line.strip()
        
        # Skip comments and empty lines
        if not line_stripped or line_stripped.startswith('#'):
            return
        
        # Check path patterns
        for pattern, pattern_type in self.path_patterns:
            matches = re.findall(pattern, line)
            for match in matches:
                self.findings.append({
                    "file": file_path,
                    "line": line_num,
                    "type": pattern_type,
                    "pattern": "hardcoded_path",
                    "match": match,
                    "line_content": line_stripped,
                    "suggestion": self._get_path_suggestion(match, pattern_type)
                })
        
        # Check file I/O patterns
        for pattern, pattern_type in self.file_io_patterns:
            if re.search(pattern, line):
                self.findings.append({
                    "file": file_path,
                    "line": line_num,
                    "type": pattern_type,
                    "pattern": "direct_file_io",
                    "match": pattern,
                    "line_content": line_stripped,
                    "suggestion": self._get_io_suggestion(pattern_type)
                })
    
    def _get_path_suggestion(self, match: str, pattern_type: str) -> str:
        """Generate suggestion for path replacement"""
        if "backup_data" in pattern_type:
            return "Use Data_Registry.get_backup_path(exchange, symbol)"
        elif "data_directory" in pattern_type:
            return "Use Data_Registry.get_data_path(branch, mode, dataset_type)"
        elif "logs_directory" in pattern_type:
            return "Use Data_Registry.get_log_path(branch, mode, log_type)"
        elif "models_directory" in pattern_type:
            return "Use Data_Registry.get_model_path(branch, model_type)"
        elif "state_directory" in pattern_type:
            return "Use Data_Registry.get_state_path(branch, mode)"
        elif "csv_file" in pattern_type:
            return "Use Data_Registry methods instead of hardcoded .csv paths"
        elif "json_file" in pattern_type:
            return "Use Data_Registry methods instead of hardcoded .json paths"
        elif "log_file" in pattern_type:
            return "Use Log_Manager for structured logging"
        else:
            return "Replace with Data_Registry method call"
    
    def _get_io_suggestion(self, pattern_type: str) -> str:
        """Generate suggestion for I/O replacement"""
        if "csv" in pattern_type:
            return "Use Data_Manager.read_csv() / write_csv() with Data_Registry paths"
        elif "json" in pattern_type:
            return "Use Data_Manager.read_json() / write_json() with Data_Registry paths"
        elif "pickle" in pattern_type:
            return "Use Save_AI_Update module for model persistence"
        elif "open" in pattern_type:
            return "Use Data_Manager methods with Data_Registry paths"
        else:
            return "Use appropriate Data_Manager method instead of direct I/O"
    
    def _generate_findings_report(self):
        """Generate detailed findings report"""
        output_path = self.root_path / "artifacts" / "hardcoded_paths_report.md"
        
        with open(output_path, 'w') as f:
            f.write("# Hardcoded Paths and Direct I/O Report\n\n")
            f.write(f"Generated: {self._get_timestamp()}\n\n")
            f.write(f"Total findings: {len(self.findings)}\n\n")
            
            # Group by pattern type
            by_type = {}
            for finding in self.findings:
                pattern_type = finding['type']
                if pattern_type not in by_type:
                    by_type[pattern_type] = []
                by_type[pattern_type].append(finding)
            
            for pattern_type, findings in sorted(by_type.items()):
                f.write(f"## {pattern_type.replace('_', ' ').title()}\n\n")
                f.write(f"Found {len(findings)} instances:\n\n")
                
                for finding in findings:
                    f.write(f"**{finding['file']}:{finding['line']}**\n")
                    f.write(f"```python\n{finding['line_content']}\n```\n")
                    f.write(f"**Suggestion:** {finding['suggestion']}\n\n")
                    
        print(f"📋 Generated {output_path}")
    
    def _generate_replacement_suggestions(self):
        """Generate path_replacements.md with specific suggestions"""
        output_path = self.root_path / "artifacts" / "path_replacements.md"
        
        with open(output_path, 'w') as f:
            f.write("# Path Replacement Suggestions\n\n")
            f.write(f"Generated: {self._get_timestamp()}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- Total hardcoded paths found: {len([f for f in self.findings if f['pattern'] == 'hardcoded_path'])}\n")
            f.write(f"- Total direct I/O operations found: {len([f for f in self.findings if f['pattern'] == 'direct_file_io'])}\n\n")
            
            f.write("## Recommended Data_Registry Methods\n\n")
            f.write("Replace hardcoded paths with these Data_Registry method calls:\n\n")
            
            registry_methods = [
                ("get_data_path(branch, mode, dataset_type)", "Historical OHLCV data, indicators, features"),
                ("get_model_path(branch, model_type)", "Trained models, artifacts, checkpoints"),
                ("get_log_path(branch, mode, log_type)", "Structured logs, error logs, decision traces"),
                ("get_backup_path(exchange, symbol)", "Historical data backups by exchange"),
                ("get_state_path(branch, mode)", "Runtime state, positions, metrics"),
                ("get_metrics_path(branch, mode)", "Performance metrics, backtesting results"),
                ("get_decisions_path(branch, mode)", "Trading decisions, signals, analysis")
            ]
            
            for method, description in registry_methods:
                f.write(f"- `Data_Registry.{method}` - {description}\n")
            
            f.write("\n## File-by-File Replacements\n\n")
            
            # Group by file
            by_file = {}
            for finding in self.findings:
                file_path = finding['file']
                if file_path not in by_file:
                    by_file[file_path] = []
                by_file[file_path].append(finding)
            
            for file_path, findings in sorted(by_file.items()):
                f.write(f"### {file_path}\n\n")
                for finding in findings:
                    f.write(f"Line {finding['line']}: {finding['suggestion']}\n")
                f.write("\n")
        
        print(f"📝 Generated {output_path}")
    
    def _get_timestamp(self):
        """Get current timestamp"""
        import time
        return time.strftime('%Y-%m-%d %H:%M:%S')


def main():
    """Main entry point"""
    root_path = Path(__file__).parent.parent
    finder = HardcodedPathFinder(str(root_path))
    
    findings = finder.scan()
    
    # Print summary
    print("\n" + "="*50)
    print("🎯 HARDCODED PATHS SUMMARY")
    print("="*50)
    
    hardcoded_paths = [f for f in findings if f['pattern'] == 'hardcoded_path']
    direct_io = [f for f in findings if f['pattern'] == 'direct_file_io']
    
    print(f"Hardcoded paths: {len(hardcoded_paths)}")
    print(f"Direct I/O operations: {len(direct_io)}")
    print(f"Total issues: {len(findings)}")
    
    if findings:
        print("\n🔧 Next steps:")
        print("1. Review artifacts/hardcoded_paths_report.md")
        print("2. Create Data_Registry.py with suggested methods")
        print("3. Update imports to use Data_Registry")
        print("4. Replace direct I/O with Data_Manager methods")
    else:
        print("\n✅ No hardcoded paths found!")
    
    print("\n📁 Reports generated in ./artifacts/")
    print("- hardcoded_paths_report.md")
    print("- path_replacements.md")


if __name__ == "__main__":
    main()