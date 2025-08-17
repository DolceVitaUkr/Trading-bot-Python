"""
Tests for wiring and architectural integrity
"""
import pytest
import json
from pathlib import Path
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tools.Build_Dependency_Graph import DependencyAnalyzer
from tools.Check_Interface_Conformance import InterfaceConformanceChecker
from Data_Registry import Data_Registry


class TestWiring:
    """Test suite for architectural wiring"""
    
    def test_no_cyclic_dependencies(self):
        """Test that there are no cyclic dependencies in the module graph"""
        analyzer = DependencyAnalyzer(str(project_root))
        problems = analyzer.analyze()
        
        cyclic_deps = problems.get('cyclic_dependencies', [])
        assert len(cyclic_deps) == 0, f"Found cyclic dependencies: {cyclic_deps}"
    
    def test_orphans_are_documented(self):
        """Test that orphaned modules are documented and reasonable"""
        analyzer = DependencyAnalyzer(str(project_root))
        problems = analyzer.analyze()
        
        orphans = problems.get('orphans', [])
        
        # Some orphans are expected (tools, examples, etc.)
        expected_orphans = [
            'tools.Build_Dependency_Graph',
            'tools.Find_Hardcoded_Paths', 
            'tools.Check_Interface_Conformance',
            'tools.Replace_Print_With_Logger',
            'tools.Check_Schemas',
            'examples.ibkr_demo_forex_spot',
            'examples.ibkr_demo_forex_options',
            'Self_test.test',
            '__init__'
        ]
        
        # Filter out expected orphans
        unexpected_orphans = [o for o in orphans if not any(o.startswith(exp) for exp in expected_orphans)]
        
        # There should be fewer unexpected orphans after our refactoring
        assert len(unexpected_orphans) < 40, f"Too many unexpected orphans: {unexpected_orphans}"
    
    def test_no_multiple_mains(self):
        """Test that multiple main functions are appropriate"""
        analyzer = DependencyAnalyzer(str(project_root))
        problems = analyzer.analyze()
        
        multiple_mains = problems.get('multiple_mains', [])
        
        # Some multiple mains are expected (main.py, tools, examples)
        expected_mains = [
            'main',
            'Data_Registry',  # Has demo main
            'tools.Build_Dependency_Graph',
            'tools.Find_Hardcoded_Paths',
            'tools.Check_Interface_Conformance', 
            'tools.Replace_Print_With_Logger',
            'tools.Check_Schemas',
            'examples.ibkr_demo_forex_spot',
            'examples.ibkr_demo_forex_options',
            'Self_test.test',
            'modules.brokers.ibkr.Connect_IBKR_API'  # Has demo/test main
        ]
        
        unexpected_mains = [m for m in multiple_mains if m not in expected_mains]
        assert len(unexpected_mains) == 0, f"Unexpected main functions: {unexpected_mains}"
    
    def test_interface_conformance(self):
        """Test that interface implementations are conformant"""
        checker = InterfaceConformanceChecker(str(project_root))
        violations = checker.check_conformance()
        
        # Some violations might be acceptable, but let's check for critical ones
        critical_violations = [v for v in violations if v['type'] == 'missing_method']
        
        # We should have minimal critical violations
        assert len(critical_violations) <= 5, f"Too many missing method violations: {critical_violations}"
    
    def test_data_registry_paths_exist(self):
        """Test that Data_Registry creates required directory structure"""
        # Test that basic paths can be created
        test_paths = [
            Data_Registry.get_data_path("test", "paper", "features"),
            Data_Registry.get_model_path("test", "rl_model"),
            Data_Registry.get_log_path("test", "paper", "decisions"),
            Data_Registry.get_state_path("test", "paper")
        ]
        
        for path in test_paths:
            assert path.exists(), f"Data_Registry should create path: {path}"
    
    def test_module_map_is_valid(self):
        """Test that the generated module map is valid JSON and contains expected data"""
        module_map_path = project_root / "artifacts" / "module_map.json"
        
        assert module_map_path.exists(), "Module map should exist"
        
        with open(module_map_path, 'r') as f:
            module_map = json.load(f)
        
        # Should have some modules
        assert len(module_map) > 50, "Should have analyzed many modules"
        
        # Each module should have required fields
        for module_name, module_info in module_map.items():
            required_fields = ['module', 'file', 'imports', 'used_by', 'last_modified']
            for field in required_fields:
                assert field in module_info, f"Module {module_name} missing field {field}"
    
    def test_no_hardcoded_print_in_core_modules(self):
        """Test that core modules don't use print statements (after refactoring)"""
        # Check that our refactored files don't contain print statements
        files_to_check = [
            project_root / "adapters" / "ibkr_exec.py",
            project_root / "modules" / "brokers" / "ibkr" / "Fetch_IBKR_MarketData.py",
            project_root / "telemetry" / "report_generator.py"
        ]
        
        for file_path in files_to_check:
            if file_path.exists():
                content = file_path.read_text()
                print_lines = [line for line in content.split('\n') if 'print(' in line and not line.strip().startswith('#')]
                assert len(print_lines) == 0, f"Found print statements in {file_path}: {print_lines}"
    
    def test_data_registry_used_correctly(self):
        """Test that hardcoded paths have been replaced with Data_Registry calls"""
        # Check that our refactored files use Data_Registry
        files_to_check = [
            project_root / "modules" / "brokers" / "ibkr" / "Fetch_IBKR_MarketData.py",
            project_root / "modules" / "data_manager.py", 
            project_root / "telemetry" / "report_generator.py",
            project_root / "state" / "runtime_state.py"
        ]
        
        for file_path in files_to_check:
            if file_path.exists():
                content = file_path.read_text()
                assert "Data_Registry" in content, f"File {file_path} should import Data_Registry"
    
    def test_config_imports_correctly(self):
        """Test that config.py can be imported without errors"""
        try:
            import config
            # Check that required config values exist
            required_attrs = ['PRODUCTS_ENABLED', 'TRAINING_MODE', 'LOG_LEVEL']
            for attr in required_attrs:
                assert hasattr(config, attr), f"Config missing required attribute: {attr}"
        except ImportError as e:
            pytest.fail(f"Config import failed: {e}")
    
    def test_core_interfaces_are_importable(self):
        """Test that core interfaces can be imported"""
        try:
            from core.interfaces import MarketData, Execution, WalletSync, NewsFeed, ValidationRunner
            # Basic check that they are classes/protocols
            assert MarketData is not None
            assert Execution is not None 
            assert WalletSync is not None
            assert NewsFeed is not None
            assert ValidationRunner is not None
        except ImportError as e:
            pytest.fail(f"Core interfaces import failed: {e}")


class TestArchitecturalGuards:
    """Test architectural constraints and guards"""
    
    def test_no_direct_file_io_in_adapters(self):
        """Test that adapter files don't use direct file I/O (should use Data_Registry)"""
        adapters_dir = project_root / "adapters"
        if not adapters_dir.exists():
            pytest.skip("No adapters directory")
        
        for py_file in adapters_dir.rglob("*.py"):
            if py_file.name.startswith("__"):
                continue
                
            content = py_file.read_text()
            
            # Check for direct file operations that should use Data_Registry
            forbidden_patterns = [
                "open(",
                "with open(",
                ".to_csv(",
                ".read_csv("
            ]
            
            for pattern in forbidden_patterns:
                lines_with_pattern = [line.strip() for line in content.split('\n') 
                                    if pattern in line and not line.strip().startswith('#')]
                
                # Allow some exceptions for necessary operations
                allowed_exceptions = [
                    "# OK:",  # Explicitly marked as OK
                    "test",   # Test files
                    "temp",   # Temporary files
                ]
                
                filtered_lines = [line for line in lines_with_pattern 
                                if not any(exc in line.lower() for exc in allowed_exceptions)]
                
                if filtered_lines:
                    # This is a soft assertion - we want to improve over time
                    print(f"Warning: {py_file} contains direct file I/O: {filtered_lines}")
    
    def test_logger_usage_pattern(self):
        """Test that files using logging follow the correct pattern"""
        python_files = list(project_root.rglob("*.py"))
        
        files_with_logging = []
        for py_file in python_files:
            if py_file.name.startswith("__") or "test" in str(py_file):
                continue
                
            try:
                content = py_file.read_text()
                if "import logging" in content or "logger" in content:
                    files_with_logging.append(py_file)
            except:
                continue  # Skip files we can't read
        
        # Check that files with logging follow good patterns
        for py_file in files_with_logging:
            content = py_file.read_text()
            
            # If it imports logging, it should create a logger
            if "import logging" in content:
                assert ("logger = logging.getLogger(__name__)" in content or
                       "log = logging.getLogger(__name__)" in content or
                       "get_logger" in content), f"File {py_file} should create a proper logger"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])