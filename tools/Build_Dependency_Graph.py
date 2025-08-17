#!/usr/bin/env python3
"""
Build_Dependency_Graph.py

Analyzes the codebase to generate:
1. module_map.json - Complete inventory of modules with metadata
2. dependency_graph.png - Visual dependency graph
3. Flags architectural problems (orphans, cycles, duplicates)
"""

import ast
import json
import os
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class DependencyAnalyzer:
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.module_map = {}
        self.import_graph = defaultdict(set)
        self.reverse_graph = defaultdict(set)
        self.all_modules = set()
        
    def analyze(self):
        """Main analysis entry point"""
        print("🔍 Starting dependency analysis...")
        
        # Find all Python files
        python_files = list(self.root_path.rglob("*.py"))
        print(f"📁 Found {len(python_files)} Python files")
        
        # Analyze each file
        for py_file in python_files:
            if self._should_skip_file(py_file):
                continue
                
            try:
                self._analyze_file(py_file)
            except Exception as e:
                print(f"⚠️  Error analyzing {py_file}: {e}")
                
        print(f"📊 Analyzed {len(self.module_map)} modules")
        
        # Find problems
        problems = self._find_problems()
        
        # Generate outputs
        self._generate_module_map()
        self._generate_dot_graph()
        self._generate_problems_report(problems)
        
        return problems
    
    def _should_skip_file(self, py_file: Path) -> bool:
        """Skip certain files from analysis"""
        skip_patterns = [
            "__pycache__",
            ".venv",
            "venv",
            ".git",
            "test_",  # Skip test files for dependency analysis
            ".pytest_cache"
        ]
        
        file_str = str(py_file)
        return any(pattern in file_str for pattern in skip_patterns)
    
    def _analyze_file(self, py_file: Path):
        """Analyze a single Python file for imports and metadata"""
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse AST
            tree = ast.parse(content, filename=str(py_file))
            
            # Get module name relative to project root
            module_name = self._get_module_name(py_file)
            
            # Extract imports
            imports = self._extract_imports(tree)
            
            # Get file metadata
            stat = py_file.stat()
            
            # Store module info
            self.module_map[module_name] = {
                "module": module_name,
                "file": str(py_file.relative_to(self.root_path)),
                "imports": list(imports),
                "used_by": [],  # Will be populated later
                "last_modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
                "size_bytes": stat.st_size,
                "line_count": len(content.splitlines()),
                "has_main": self._has_main_function(tree),
                "is_executable": self._is_executable_script(tree)
            }
            
            # Build graph
            self.all_modules.add(module_name)
            for imp in imports:
                if self._is_local_import(imp):
                    self.import_graph[module_name].add(imp)
                    self.reverse_graph[imp].add(module_name)
                    
        except Exception as e:
            print(f"❌ Failed to analyze {py_file}: {e}")
    
    def _get_module_name(self, py_file: Path) -> str:
        """Convert file path to module name"""
        rel_path = py_file.relative_to(self.root_path)
        
        # Remove .py extension
        if rel_path.name == "__init__.py":
            # For __init__.py, use parent directory
            parts = rel_path.parent.parts
        else:
            parts = rel_path.with_suffix("").parts
            
        return ".".join(parts) if parts else py_file.stem
    
    def _extract_imports(self, tree: ast.AST) -> Set[str]:
        """Extract import statements from AST"""
        imports = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
                    # Also add specific imported names if from local modules
                    if self._is_local_import(node.module):
                        for alias in node.names:
                            if alias.name != "*":
                                full_name = f"{node.module}.{alias.name}"
                                imports.add(full_name)
                                
        return imports
    
    def _is_local_import(self, import_name: str) -> bool:
        """Check if import is from local project (not external library)"""
        if not import_name:
            return False
            
        # Check if it starts with known local modules
        local_prefixes = [
            "core", "adapters", "managers", "modules", "utils", 
            "config", "main", "forex", "state", "telemetry",
            "options", "policies"
        ]
        
        return any(import_name.startswith(prefix) for prefix in local_prefixes)
    
    def _has_main_function(self, tree: ast.AST) -> bool:
        """Check if file has a main function"""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "main":
                return True
            if isinstance(node, ast.If):
                # Check for if __name__ == "__main__"
                if (isinstance(node.test, ast.Compare) and
                    isinstance(node.test.left, ast.Name) and
                    node.test.left.id == "__name__"):
                    return True
        return False
    
    def _is_executable_script(self, tree: ast.AST) -> bool:
        """Check if file is an executable script"""
        # Look for if __name__ == "__main__" pattern
        for node in ast.walk(tree):
            if (isinstance(node, ast.If) and
                isinstance(node.test, ast.Compare) and
                isinstance(node.test.left, ast.Name) and
                node.test.left.id == "__name__"):
                return True
        return False
    
    def _find_problems(self) -> Dict[str, List[str]]:
        """Find architectural problems"""
        problems = {
            "orphans": [],
            "leaf_executables": [],
            "cyclic_dependencies": [],
            "multiple_mains": [],
            "duplicate_modules": []
        }
        
        # Find orphans (no one imports them and they're not executables)
        for module in self.all_modules:
            if (module not in self.reverse_graph and 
                not self.module_map[module].get("is_executable", False)):
                problems["orphans"].append(module)
        
        # Find leaf executables (scripts with no callers)
        for module in self.all_modules:
            module_info = self.module_map[module]
            if (module_info.get("is_executable", False) and 
                module not in self.reverse_graph):
                problems["leaf_executables"].append(module)
        
        # Find cyclic dependencies
        cycles = self._find_cycles()
        problems["cyclic_dependencies"] = cycles
        
        # Find multiple main functions
        main_modules = [mod for mod, info in self.module_map.items() 
                       if info.get("has_main", False)]
        if len(main_modules) > 1:
            problems["multiple_mains"] = main_modules
        
        # Find potential duplicates (by similar names or purpose)
        problems["duplicate_modules"] = self._find_duplicate_modules()
        
        return problems
    
    def _find_cycles(self) -> List[List[str]]:
        """Find cyclic dependencies using DFS"""
        visited = set()
        rec_stack = set()
        cycles = []
        
        def dfs(node, path):
            if node in rec_stack:
                # Found a cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return
                
            if node in visited:
                return
                
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in self.import_graph[node]:
                if neighbor in self.all_modules:  # Only consider local modules
                    dfs(neighbor, path + [node])
            
            rec_stack.remove(node)
        
        for module in self.all_modules:
            if module not in visited:
                dfs(module, [])
                
        return cycles
    
    def _find_duplicate_modules(self) -> List[Tuple[str, str, str]]:
        """Find potentially duplicate modules"""
        duplicates = []
        modules = list(self.module_map.keys())
        
        for i, mod1 in enumerate(modules):
            for mod2 in modules[i+1:]:
                # Check for similar names
                name1 = mod1.split(".")[-1].lower()
                name2 = mod2.split(".")[-1].lower()
                
                # Simple similarity check
                if (name1 in name2 or name2 in name1) and abs(len(name1) - len(name2)) <= 2:
                    duplicates.append((mod1, mod2, "similar_name"))
                    
        return duplicates
    
    def _generate_module_map(self):
        """Generate module_map.json with used_by populated"""
        # Populate used_by fields
        for module, importers in self.reverse_graph.items():
            if module in self.module_map:
                self.module_map[module]["used_by"] = list(importers)
        
        output_path = self.root_path / "artifacts" / "module_map.json"
        with open(output_path, 'w') as f:
            json.dump(self.module_map, f, indent=2, sort_keys=True)
        
        print(f"📝 Generated {output_path}")
    
    def _generate_dot_graph(self):
        """Generate dependency graph in DOT format and convert to PNG"""
        dot_content = "digraph dependencies {\n"
        dot_content += "  rankdir=LR;\n"
        dot_content += "  node [shape=box];\n\n"
        
        # Add nodes with colors based on type
        for module in self.all_modules:
            info = self.module_map[module]
            color = "lightblue"
            
            if info.get("is_executable"):
                color = "lightgreen"
            elif module in self.reverse_graph and len(self.reverse_graph[module]) == 0:
                color = "lightyellow"  # Orphan
                
            dot_content += f'  "{module}" [fillcolor={color}, style=filled];\n'
        
        # Add edges
        for module, imports in self.import_graph.items():
            for imp in imports:
                if imp in self.all_modules:
                    dot_content += f'  "{module}" -> "{imp}";\n'
        
        dot_content += "}\n"
        
        # Write DOT file
        dot_path = self.root_path / "artifacts" / "dependency_graph.dot"
        with open(dot_path, 'w') as f:
            f.write(dot_content)
        
        # Try to generate PNG if graphviz is available
        try:
            import subprocess
            png_path = self.root_path / "artifacts" / "dependency_graph.png"
            subprocess.run(["dot", "-Tpng", str(dot_path), "-o", str(png_path)], 
                         check=True, capture_output=True)
            print(f"🖼️  Generated {png_path}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"⚠️  Could not generate PNG (graphviz not available). DOT file: {dot_path}")
    
    def _generate_problems_report(self, problems: Dict[str, List]):
        """Generate problems report"""
        output_path = self.root_path / "artifacts" / "architectural_problems.md"
        
        with open(output_path, 'w') as f:
            f.write("# Architectural Problems Report\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for problem_type, items in problems.items():
                f.write(f"## {problem_type.replace('_', ' ').title()}\n\n")
                
                if not items:
                    f.write("✅ No issues found.\n\n")
                else:
                    for item in items:
                        if isinstance(item, list):
                            f.write(f"- Cycle: {' -> '.join(item)}\n")
                        elif isinstance(item, tuple):
                            f.write(f"- {item[0]} ↔ {item[1]} ({item[2]})\n")
                        else:
                            f.write(f"- {item}\n")
                    f.write("\n")
        
        print(f"📋 Generated {output_path}")


def main():
    """Main entry point"""
    root_path = Path(__file__).parent.parent
    analyzer = DependencyAnalyzer(str(root_path))
    
    problems = analyzer.analyze()
    
    # Print summary
    print("\n" + "="*50)
    print("🎯 ANALYSIS SUMMARY")
    print("="*50)
    print(f"Total modules: {len(analyzer.module_map)}")
    print(f"Import relationships: {sum(len(imports) for imports in analyzer.import_graph.values())}")
    
    for problem_type, items in problems.items():
        if items:
            print(f"❌ {problem_type.replace('_', ' ').title()}: {len(items)}")
        else:
            print(f"✅ {problem_type.replace('_', ' ').title()}: 0")
    
    print("\n📁 Artifacts generated in ./artifacts/")
    print("- module_map.json")
    print("- dependency_graph.dot")
    print("- dependency_graph.png (if graphviz available)")
    print("- architectural_problems.md")


if __name__ == "__main__":
    main()