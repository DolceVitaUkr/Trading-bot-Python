#!/usr/bin/env python3
"""
Replace_Print_With_Logger.py

Safely replaces print() statements with appropriate logger calls.
Analyzes context to determine appropriate log level (info, debug, warning, error).
"""

import ast
import re
from pathlib import Path
from typing import List, Dict, Tuple
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class PrintReplacer:
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.replacements = []
        self.dry_run = True
        
    def replace_prints(self, dry_run: bool = True):
        """Main entry point for replacing print statements"""
        self.dry_run = dry_run
        print(f"🔍 {'Analyzing' if dry_run else 'Replacing'} print statements...")
        
        python_files = list(self.root_path.rglob("*.py"))
        skipped_files = []
        
        for py_file in python_files:
            if self._should_skip_file(py_file):
                skipped_files.append(str(py_file.relative_to(self.root_path)))
                continue
                
            try:
                self._process_file(py_file)
            except Exception as e:
                print(f"⚠️  Error processing {py_file}: {e}")
        
        print(f"📊 Processed {len(python_files) - len(skipped_files)} files")
        print(f"📋 Found {len(self.replacements)} print statements")
        
        if skipped_files:
            print(f"⏭️  Skipped {len(skipped_files)} files (tests, tools, etc.)")
        
        # Generate report
        self._generate_replacement_report()
        
        return self.replacements
    
    def _should_skip_file(self, py_file: Path) -> bool:
        """Skip certain files from replacement"""
        skip_patterns = [
            "__pycache__",
            ".venv", "venv",
            ".git",
            "test_",  # Skip test files
            ".pytest_cache",
            "tools/Replace_Print_With_Logger.py",  # Skip this file itself
            "tools/",  # Skip all tools for now
            "examples/"  # Skip example files
        ]
        
        file_str = str(py_file)
        return any(pattern in file_str for pattern in skip_patterns)
    
    def _process_file(self, py_file: Path):
        """Process a single file for print replacements"""
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_lines = content.splitlines()
            
            # Parse AST to find print calls
            try:
                tree = ast.parse(content, filename=str(py_file))
                print_calls = self._find_print_calls(tree)
            except SyntaxError:
                print(f"⚠️  Syntax error in {py_file}, skipping")
                return
            
            if not print_calls:
                return
            
            # Analyze each print call
            modifications = []
            for call_info in print_calls:
                replacement = self._analyze_print_call(py_file, original_lines, call_info)
                if replacement:
                    modifications.append(replacement)
            
            if modifications and not self.dry_run:
                # Apply modifications (implement later for safety)
                pass
            
            self.replacements.extend(modifications)
            
        except Exception as e:
            print(f"❌ Failed to process {py_file}: {e}")
    
    def _find_print_calls(self, tree: ast.AST) -> List[Dict]:
        """Find all print() calls in the AST"""
        print_calls = []
        
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and 
                isinstance(node.func, ast.Name) and 
                node.func.id == 'print'):
                
                print_calls.append({
                    'line': node.lineno,
                    'col': node.col_offset,
                    'args': [self._ast_to_string(arg) for arg in node.args],
                    'keywords': {kw.arg: self._ast_to_string(kw.value) for kw in node.keywords if kw.arg}
                })
        
        return print_calls
    
    def _ast_to_string(self, node: ast.AST) -> str:
        """Convert AST node back to string representation"""
        if isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._ast_to_string(node.value)}.{node.attr}"
        elif isinstance(node, ast.Call):
            func_name = self._ast_to_string(node.func)
            args = [self._ast_to_string(arg) for arg in node.args]
            return f"{func_name}({', '.join(args)})"
        elif isinstance(node, ast.BinOp):
            left = self._ast_to_string(node.left)
            right = self._ast_to_string(node.right)
            op = self._get_operator(node.op)
            return f"{left} {op} {right}"
        elif isinstance(node, ast.JoinedStr):  # f-strings
            return "f'...'"
        else:
            return "..."
    
    def _get_operator(self, op: ast.operator) -> str:
        """Get string representation of operator"""
        ops = {
            ast.Add: '+',
            ast.Sub: '-',
            ast.Mult: '*',
            ast.Div: '/',
            ast.Mod: '%'
        }
        return ops.get(type(op), '?')
    
    def _analyze_print_call(self, py_file: Path, lines: List[str], call_info: Dict) -> Dict:
        """Analyze a print call and determine appropriate replacement"""
        line_num = call_info['line']
        line_content = lines[line_num - 1].strip() if line_num <= len(lines) else ""
        
        # Determine log level based on context
        log_level = self._determine_log_level(line_content, call_info['args'])
        
        # Generate logger import if needed
        needs_logger_import = not self._has_logger_import(lines)
        
        # Generate replacement
        logger_call = self._generate_logger_call(call_info, log_level)
        
        replacement = {
            'file': str(py_file.relative_to(self.root_path)),
            'line': line_num,
            'original': line_content,
            'replacement': logger_call,
            'log_level': log_level,
            'needs_logger_import': needs_logger_import,
            'args': call_info['args']
        }
        
        return replacement
    
    def _determine_log_level(self, line_content: str, args: List[str]) -> str:
        """Determine appropriate log level based on content"""
        line_lower = line_content.lower()
        args_text = ' '.join(args).lower() if args else ""
        combined = f"{line_lower} {args_text}"
        
        # Error patterns
        if any(pattern in combined for pattern in [
            'error', 'failed', 'exception', 'traceback', 
            'cannot', 'unable', 'invalid', 'missing'
        ]):
            return 'error'
        
        # Warning patterns
        if any(pattern in combined for pattern in [
            'warning', 'warn', 'deprecated', 'skip', 
            'fallback', 'retry', 'timeout'
        ]):
            return 'warning'
        
        # Debug patterns
        if any(pattern in combined for pattern in [
            'debug', 'trace', 'dump', 'analyzing', 
            'checking', 'found', 'processing'
        ]):
            return 'debug'
        
        # Success/status patterns
        if any(pattern in combined for pattern in [
            'success', 'completed', 'generated', 'connected',
            'started', 'finished', 'loaded'
        ]):
            return 'info'
        
        # Default to info for general output
        return 'info'
    
    def _has_logger_import(self, lines: List[str]) -> bool:
        """Check if file already has logger import"""
        content = '\n'.join(lines[:50])  # Check first 50 lines
        return any(pattern in content for pattern in [
            'from modules.Logger_Config import',
            'import logging',
            'logger =',
            'log =',
            'get_logger'
        ])
    
    def _generate_logger_call(self, call_info: Dict, log_level: str) -> str:
        """Generate appropriate logger call"""
        args = call_info['args']
        keywords = call_info.get('keywords', {})
        
        if len(args) == 1:
            # Simple case: print("message")
            message = args[0]
            return f"logger.{log_level}({message})"
        elif len(args) > 1:
            # Multiple args: print("msg", var1, var2)
            # Convert to f-string or formatted message
            first_arg = args[0]
            if first_arg.startswith('"') and '%s' in first_arg:
                # String formatting: print("msg %s", var)
                return f"logger.{log_level}({first_arg}, {', '.join(args[1:])})"
            else:
                # Multiple separate args - combine them
                combined_args = ', '.join(args)
                return f"logger.{log_level}(f\"{{{combined_args}}}\")"
        else:
            # No args: print()
            return f"logger.{log_level}(\"\")"
    
    def _generate_replacement_report(self):
        """Generate replacement report"""
        output_path = self.root_path / "artifacts" / "print_replacements.md"
        
        with open(output_path, 'w') as f:
            f.write("# Print Statement Replacement Report\n\n")
            f.write(f"Generated: {self._get_timestamp()}\n")
            f.write(f"Mode: {'Analysis' if self.dry_run else 'Applied'}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"Total print statements found: {len(self.replacements)}\n")
            
            # Group by log level
            by_level = {}
            for repl in self.replacements:
                level = repl['log_level']
                by_level[level] = by_level.get(level, 0) + 1
            
            f.write("\n**By log level:**\n")
            for level, count in sorted(by_level.items()):
                f.write(f"- {level}: {count}\n")
            
            # Group by file
            by_file = {}
            for repl in self.replacements:
                file_path = repl['file']
                if file_path not in by_file:
                    by_file[file_path] = []
                by_file[file_path].append(repl)
            
            f.write(f"\nFiles affected: {len(by_file)}\n\n")
            
            # Detailed replacements
            f.write("## Detailed Replacements\n\n")
            
            for file_path, replacements in sorted(by_file.items()):
                f.write(f"### {file_path}\n\n")
                
                if replacements[0]['needs_logger_import']:
                    f.write("**Add import:**\n")
                    f.write("```python\n")
                    f.write("from modules.Logger_Config import get_logger\n")
                    f.write("logger = get_logger(__name__)\n")
                    f.write("```\n\n")
                
                f.write("**Replacements:**\n\n")
                for repl in replacements:
                    f.write(f"**Line {repl['line']}** ({repl['log_level']} level):\n")
                    f.write("```python\n")
                    f.write(f"# Before:\n{repl['original']}\n")
                    f.write(f"# After:\n{repl['replacement']}\n")
                    f.write("```\n\n")
        
        print(f"📋 Generated {output_path}")
    
    def _get_timestamp(self):
        """Get current timestamp"""
        import time
        return time.strftime('%Y-%m-%d %H:%M:%S')


def main():
    """Main entry point"""
    root_path = Path(__file__).parent.parent
    replacer = PrintReplacer(str(root_path))
    
    # Always run in dry run mode for safety
    replacements = replacer.replace_prints(dry_run=True)
    
    # Print summary
    print("\n" + "="*50)
    print("🎯 PRINT REPLACEMENT SUMMARY")
    print("="*50)
    print(f"Print statements found: {len(replacements)}")
    
    # Group by log level
    by_level = {}
    for repl in replacements:
        level = repl['log_level']
        by_level[level] = by_level.get(level, 0) + 1
    
    print("\nBy log level:")
    for level, count in sorted(by_level.items()):
        print(f"- {level}: {count}")
    
    # Group by file
    by_file = {}
    for repl in replacements:
        file_path = repl['file']
        by_file[file_path] = by_file.get(file_path, 0) + 1
    
    print(f"\nFiles affected: {len(by_file)}")
    if len(by_file) <= 10:
        for file_path, count in sorted(by_file.items(), key=lambda x: x[1], reverse=True):
            print(f"- {file_path}: {count}")
    
    print("\n📋 Detailed report: ./artifacts/print_replacements.md")
    print("⚠️  Run with --apply flag to actually apply changes (not implemented for safety)")


if __name__ == "__main__":
    main()