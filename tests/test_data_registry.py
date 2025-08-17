"""
Tests for Data_Registry functionality
"""
import pytest
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from Data_Registry import DataRegistry


class TestDataRegistry:
    """Test suite for Data_Registry"""
    
    @pytest.fixture
    def temp_registry(self):
        """Create a temporary registry for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set temporary paths
            temp_path = Path(temp_dir)
            registry = DataRegistry()
            
            # Override paths to use temp directory
            registry.data_root = temp_path / "data"
            registry.backup_root = temp_path / "backup_data"
            registry.models_root = temp_path / "models"
            registry.logs_root = temp_path / "logs"
            registry.state_root = temp_path / "state"
            
            # Create base directories
            registry._ensure_base_directories()
            
            yield registry
    
    def test_data_registry_initialization(self, temp_registry):
        """Test that DataRegistry initializes correctly"""
        registry = temp_registry
        assert registry.data_root.exists()
        assert registry.backup_root.exists()
        assert registry.models_root.exists()
        assert registry.logs_root.exists()
        assert registry.state_root.exists()
    
    def test_historical_data_paths(self, temp_registry):
        """Test historical data path generation"""
        registry = temp_registry
        
        path = registry.get_historical_data_path("bybit", "BTCUSDT", "5m")
        assert str(path).endswith("bybit/5m/BTCUSDT.csv")
        assert path.parent.exists()
        
        meta_path = registry.get_historical_meta_path("bybit", "BTCUSDT", "5m")
        assert str(meta_path).endswith("bybit/5m/BTCUSDT.meta.json")
    
    def test_branch_scoped_paths(self, temp_registry):
        """Test branch and mode scoped paths"""
        registry = temp_registry
        
        data_path = registry.get_data_path("main", "paper", "features")
        assert "data/features/main/paper" in str(data_path)
        assert data_path.exists()
        
        metrics_path = registry.get_metrics_path("experimental", "live", "backtest_results.json")
        assert "data/metrics/experimental/live/backtest_results.json" in str(metrics_path)
        assert metrics_path.parent.exists()
    
    def test_model_paths(self, temp_registry):
        """Test model path generation"""
        registry = temp_registry
        
        model_path = registry.get_model_path("main", "rl_model")
        assert "models/main/rl_model" in str(model_path)
        assert model_path.exists()
        
        checkpoint_path = registry.get_model_checkpoint_path("main", "rl_model", "epoch_100.pkl")
        assert "models/main/rl_model/checkpoints/epoch_100.pkl" in str(checkpoint_path)
        assert checkpoint_path.parent.exists()
    
    def test_log_paths(self, temp_registry):
        """Test log path generation"""
        registry = temp_registry
        
        log_path = registry.get_log_path("main", "paper", "decisions")
        assert "logs/main/paper/decisions" in str(log_path)
        assert log_path.exists()
        
        error_log = registry.get_error_log_path("main", "error.log")
        assert "logs/errors/main/error.log" in str(error_log)
        assert error_log.parent.exists()
    
    def test_state_paths(self, temp_registry):
        """Test state path generation"""
        registry = temp_registry
        
        state_path = registry.get_state_path("main", "paper")
        assert "state/main/paper" in str(state_path)
        assert state_path.exists()
        
        position_path = registry.get_position_state_path("main", "paper")
        assert "state/main/paper/positions.jsonl" in str(position_path)
    
    def test_symbol_sanitization(self, temp_registry):
        """Test symbol sanitization for filesystem safety"""
        registry = temp_registry
        
        # Test symbols with problematic characters
        path1 = registry.get_historical_data_path("bybit", "BTC/USDT", "5m")
        assert "BTCUSDT.csv" in str(path1)
        
        path2 = registry.get_historical_data_path("bybit", "EUR:USD", "1h")
        assert "EURUSD.csv" in str(path2)
    
    def test_specialized_paths(self, temp_registry):
        """Test specialized path methods"""
        registry = temp_registry
        
        ibkr_cache = registry.get_ibkr_cache_path("historical")
        assert "data/ibkr/historical" in str(ibkr_cache)
        assert ibkr_cache.exists()
        
        news_cache = registry.get_news_cache_path("rss")
        assert "data/news/rss" in str(news_cache)
        assert news_cache.exists()
        
        telegram_state = registry.get_telegram_state_path("bot_state.json")
        assert "state/telegram/bot_state.json" in str(telegram_state)
        assert telegram_state.parent.exists()
    
    def test_validation_paths(self, temp_registry):
        """Test validation result paths"""
        registry = temp_registry
        
        validation_path = registry.get_validation_results_path("main", "strategy_001")
        assert "data/validation/main/strategy_001" in str(validation_path)
        assert validation_path.exists()
    
    def test_disk_usage_report(self, temp_registry):
        """Test disk usage reporting"""
        registry = temp_registry
        
        # Create some test files
        test_dir = registry.get_data_path("main", "paper", "test_data")
        test_file_path = test_dir / "test.txt"
        test_file_path.write_text("test data")
        
        report = registry.get_disk_usage_report()
        assert "data" in report
        assert "models" in report
        assert "logs" in report
        assert "state" in report
        
        # Should have some data now
        assert report["data"]["file_count"] > 0
        assert report["data"]["size_bytes"] > 0
    
    def test_branch_list(self, temp_registry):
        """Test branch list generation"""
        registry = temp_registry
        
        # Create some branch directories
        registry.get_model_path("main", "test_model")
        registry.get_model_path("experimental", "test_model")
        registry.get_metrics_path("feature_branch", "paper")
        
        branches = registry.get_branch_list()
        assert "main" in branches
        assert "experimental" in branches
        assert "feature_branch" in branches


if __name__ == "__main__":
    pytest.main([__file__])