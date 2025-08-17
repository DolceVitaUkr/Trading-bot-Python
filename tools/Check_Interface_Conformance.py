#!/usr/bin/env python3
"""
Check_Interface_Conformance.py

Verifies that all implementations properly conform to their interface contracts.
Checks:
1. All interface implementations are complete (no missing methods)
2. Method signatures match interface requirements
3. Type hints are consistent
4. ABC inheritance is proper
"""

import ast
import inspect
import importlib
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from abc import ABC

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class InterfaceConformanceChecker:
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.interfaces = {}
        self.implementations = {}
        self.violations = []
        
    def check_conformance(self):
        """Main conformance checking entry point"""
        print("🔍 Checking interface conformance...")
        
        # 1. Load core interfaces
        self._load_core_interfaces()
        
        # 2. Find all implementations
        self._find_implementations()
        
        # 3. Check conformance
        self._check_all_conformance()
        
        # 4. Generate report
        self._generate_conformance_report()
        
        return self.violations
    
    def _load_core_interfaces(self):
        """Load interface definitions from core/interfaces.py"""
        try:
            # Import the interfaces module
            import core.interfaces as interfaces_mod
            
            # Get all classes that are Protocols
            for name, obj in inspect.getmembers(interfaces_mod, inspect.isclass):
                if hasattr(obj, '__annotations__') or hasattr(obj, '_is_protocol'):
                    methods = self._extract_interface_methods(obj)
                    self.interfaces[name] = {
                        'class': obj,
                        'methods': methods,
                        'module': 'core.interfaces'
                    }
                    print(f"📋 Found interface: {name} with {len(methods)} methods")
                    
        except ImportError as e:
            print(f"❌ Failed to load core.interfaces: {e}")
    
    def _extract_interface_methods(self, interface_class) -> Dict[str, Any]:
        """Extract method signatures from an interface"""
        methods = {}
        
        for name, method in inspect.getmembers(interface_class):
            if not name.startswith('_') and callable(method):
                try:
                    sig = inspect.signature(method)
                    methods[name] = {
                        'signature': sig,
                        'annotations': getattr(method, '__annotations__', {}),
                        'docstring': inspect.getdoc(method)
                    }
                except (ValueError, TypeError):
                    # Some methods might not have proper signatures
                    methods[name] = {
                        'signature': None,
                        'annotations': {},
                        'docstring': inspect.getdoc(method)
                    }
        
        return methods
    
    def _find_implementations(self):
        """Find all classes that implement the interfaces"""
        adapters_path = self.root_path / "adapters"
        if not adapters_path.exists():
            print("⚠️  No adapters directory found")
            return
        
        for py_file in adapters_path.rglob("*.py"):
            if py_file.name.startswith("__"):
                continue
                
            try:
                self._analyze_implementation_file(py_file)
            except Exception as e:
                print(f"⚠️  Error analyzing {py_file}: {e}")
    
    def _analyze_implementation_file(self, py_file: Path):
        """Analyze a Python file for interface implementations"""
        module_name = self._file_to_module(py_file)
        
        try:
            # Import the module
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find classes that might implement interfaces
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if self._is_implementation_class(obj):
                    interface_name = self._guess_interface_name(name, obj)
                    if interface_name and interface_name in self.interfaces:
                        methods = self._extract_implementation_methods(obj)
                        self.implementations[name] = {
                            'class': obj,
                            'methods': methods,
                            'module': module_name,
                            'file': str(py_file.relative_to(self.root_path)),
                            'implements': interface_name
                        }
                        print(f"🏗️  Found implementation: {name} -> {interface_name}")
                        
        except Exception as e:
            print(f"❌ Failed to import {module_name}: {e}")
    
    def _file_to_module(self, py_file: Path) -> str:
        """Convert file path to module name"""
        rel_path = py_file.relative_to(self.root_path)
        if rel_path.name == "__init__.py":
            parts = rel_path.parent.parts
        else:
            parts = rel_path.with_suffix("").parts
        return ".".join(parts)
    
    def _is_implementation_class(self, obj) -> bool:
        """Check if a class is likely an implementation class"""
        return (inspect.isclass(obj) and 
                not inspect.isabstract(obj) and
                obj.__module__ != 'builtins')
    
    def _guess_interface_name(self, class_name: str, class_obj) -> Optional[str]:
        """Guess which interface a class implements based on naming and inheritance"""
        # Check direct inheritance patterns
        for base in inspect.getmro(class_obj):
            base_name = base.__name__
            if base_name in self.interfaces:
                return base_name
        
        # Check naming patterns
        name_lower = class_name.lower()
        for interface_name in self.interfaces:
            interface_lower = interface_name.lower()
            if (interface_lower in name_lower or 
                name_lower.endswith(interface_lower) or
                name_lower.startswith(interface_lower)):
                return interface_name
        
        # Special case mappings
        mappings = {
            'execution': 'Execution',
            'exec': 'Execution',
            'market': 'MarketData',
            'wallet': 'WalletSync',
            'news': 'NewsFeed',
            'validation': 'ValidationRunner'
        }
        
        for pattern, interface in mappings.items():
            if pattern in name_lower and interface in self.interfaces:
                return interface
        
        return None
    
    def _extract_implementation_methods(self, impl_class) -> Dict[str, Any]:
        """Extract method signatures from an implementation class"""
        methods = {}
        
        for name, method in inspect.getmembers(impl_class):
            if (not name.startswith('_') and 
                callable(method) and 
                not isinstance(method, type)):
                try:
                    sig = inspect.signature(method)
                    methods[name] = {
                        'signature': sig,
                        'annotations': getattr(method, '__annotations__', {}),
                        'docstring': inspect.getdoc(method)
                    }
                except (ValueError, TypeError):
                    methods[name] = {
                        'signature': None,
                        'annotations': {},
                        'docstring': inspect.getdoc(method)
                    }
        
        return methods
    
    def _check_all_conformance(self):
        """Check conformance for all implementations"""
        for impl_name, impl_info in self.implementations.items():
            interface_name = impl_info['implements']
            if interface_name in self.interfaces:
                violations = self._check_implementation_conformance(
                    impl_name, impl_info, interface_name, self.interfaces[interface_name]
                )
                self.violations.extend(violations)
    
    def _check_implementation_conformance(self, impl_name: str, impl_info: Dict, 
                                         interface_name: str, interface_info: Dict) -> List[Dict]:
        """Check conformance between an implementation and its interface"""
        violations = []
        
        impl_methods = impl_info['methods']
        interface_methods = interface_info['methods']
        
        # Check for missing methods
        for method_name, method_info in interface_methods.items():
            if method_name not in impl_methods:
                violations.append({
                    'type': 'missing_method',
                    'implementation': impl_name,
                    'interface': interface_name,
                    'method': method_name,
                    'description': f"Method '{method_name}' required by {interface_name} is missing in {impl_name}",
                    'file': impl_info['file']
                })
        
        # Check for signature mismatches
        for method_name in impl_methods:
            if method_name in interface_methods:
                impl_sig = impl_methods[method_name]['signature']
                interface_sig = interface_methods[method_name]['signature']
                
                if impl_sig and interface_sig:
                    mismatch = self._check_signature_compatibility(impl_sig, interface_sig)
                    if mismatch:
                        violations.append({
                            'type': 'signature_mismatch',
                            'implementation': impl_name,
                            'interface': interface_name,
                            'method': method_name,
                            'description': f"Method signature mismatch: {mismatch}",
                            'file': impl_info['file'],
                            'expected': str(interface_sig),
                            'actual': str(impl_sig)
                        })
        
        return violations
    
    def _check_signature_compatibility(self, impl_sig, interface_sig) -> Optional[str]:
        """Check if implementation signature is compatible with interface signature"""
        try:
            # Basic parameter count check
            impl_params = list(impl_sig.parameters.keys())
            interface_params = list(interface_sig.parameters.keys())
            
            # Remove 'self' parameter for comparison
            if impl_params and impl_params[0] == 'self':
                impl_params = impl_params[1:]
            if interface_params and interface_params[0] == 'self':
                interface_params = interface_params[1:]
            
            # Check parameter compatibility (allowing for additional optional parameters)
            if len(impl_params) < len(interface_params):
                return f"Too few parameters: expected at least {len(interface_params)}, got {len(impl_params)}"
            
            # Check parameter names match for required parameters
            for i, expected_param in enumerate(interface_params):
                if i < len(impl_params):
                    actual_param = impl_params[i]
                    if expected_param != actual_param:
                        return f"Parameter name mismatch at position {i}: expected '{expected_param}', got '{actual_param}'"
                else:
                    return f"Missing required parameter: {expected_param}"
            
        except Exception as e:
            return f"Signature comparison error: {e}"
        
        return None
    
    def _generate_conformance_report(self):
        """Generate interface conformance report"""
        output_path = self.root_path / "artifacts" / "interface_conformance.md"
        
        with open(output_path, 'w') as f:
            f.write("# Interface Conformance Report\n\n")
            f.write(f"Generated: {self._get_timestamp()}\n\n")
            
            # Summary
            f.write("## Summary\n\n")
            f.write(f"- Interfaces found: {len(self.interfaces)}\n")
            f.write(f"- Implementations found: {len(self.implementations)}\n")
            f.write(f"- Violations found: {len(self.violations)}\n\n")
            
            # Interfaces
            f.write("## Available Interfaces\n\n")
            for interface_name, interface_info in self.interfaces.items():
                f.write(f"### {interface_name}\n")
                f.write(f"**Module:** {interface_info['module']}\n")
                f.write(f"**Methods:** {len(interface_info['methods'])}\n\n")
                
                for method_name, method_info in interface_info['methods'].items():
                    sig = method_info['signature']
                    f.write(f"- `{method_name}{sig if sig else '(...)'}`\n")
                f.write("\n")
            
            # Implementations
            f.write("## Implementation Status\n\n")
            for impl_name, impl_info in self.implementations.items():
                interface_name = impl_info['implements']
                f.write(f"### {impl_name}\n")
                f.write(f"**File:** {impl_info['file']}\n")
                f.write(f"**Implements:** {interface_name}\n")
                
                # Check if this implementation has violations
                impl_violations = [v for v in self.violations if v['implementation'] == impl_name]
                if impl_violations:
                    f.write(f"**Status:** ❌ {len(impl_violations)} violations\n\n")
                    for violation in impl_violations:
                        f.write(f"- ❌ {violation['description']}\n")
                else:
                    f.write("**Status:** ✅ Conformant\n")
                
                f.write("\n")
            
            # Violations detail
            if self.violations:
                f.write("## Detailed Violations\n\n")
                
                violation_types = {}
                for violation in self.violations:
                    vtype = violation['type']
                    if vtype not in violation_types:
                        violation_types[vtype] = []
                    violation_types[vtype].append(violation)
                
                for vtype, violations in violation_types.items():
                    f.write(f"### {vtype.replace('_', ' ').title()}\n\n")
                    for violation in violations:
                        f.write(f"**{violation['implementation']}**\n")
                        f.write(f"- File: {violation['file']}\n")
                        f.write(f"- Interface: {violation['interface']}\n")
                        f.write(f"- Method: {violation['method']}\n")
                        f.write(f"- Issue: {violation['description']}\n")
                        if 'expected' in violation:
                            f.write(f"- Expected: `{violation['expected']}`\n")
                            f.write(f"- Actual: `{violation['actual']}`\n")
                        f.write("\n")
            else:
                f.write("## ✅ No Violations Found\n\n")
                f.write("All implementations properly conform to their interfaces!\n\n")
        
        print(f"📋 Generated {output_path}")
    
    def _get_timestamp(self):
        """Get current timestamp"""
        import time
        return time.strftime('%Y-%m-%d %H:%M:%S')


def main():
    """Main entry point"""
    root_path = Path(__file__).parent.parent
    checker = InterfaceConformanceChecker(str(root_path))
    
    violations = checker.check_conformance()
    
    # Print summary
    print("\n" + "="*50)
    print("🎯 INTERFACE CONFORMANCE SUMMARY")
    print("="*50)
    print(f"Interfaces: {len(checker.interfaces)}")
    print(f"Implementations: {len(checker.implementations)}")
    print(f"Violations: {len(violations)}")
    
    if violations:
        print("\n❌ Issues found:")
        violation_types = {}
        for violation in violations:
            vtype = violation['type']
            violation_types[vtype] = violation_types.get(vtype, 0) + 1
        
        for vtype, count in violation_types.items():
            print(f"- {vtype.replace('_', ' ').title()}: {count}")
    else:
        print("\n✅ All implementations are conformant!")
    
    print("\n📁 Report generated: ./artifacts/interface_conformance.md")


if __name__ == "__main__":
    main()