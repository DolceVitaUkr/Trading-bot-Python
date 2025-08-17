"""
Data_Registry.py

Central registry for all file paths and data storage locations.
Single source of truth for data organization across branches and modes.

Replaces all hardcoded paths throughout the codebase.
Ensures consistent data structure: data/{category}/{branch}/{mode}/...
"""

import os
from pathlib import Path
from typing import Optional, Union
import config


class DataRegistry:
    """Central registry for all data paths and storage locations"""
    
    def __init__(self):
        self.root_path = Path.cwd()
        
        # Base directories (configurable via environment)
        self.data_root = Path(config.HISTORICAL_DATA_PATH if hasattr(config, 'HISTORICAL_DATA_PATH') else "data")
        self.backup_root = Path(os.getenv("BACKUP_DATA_PATH", "backup_data"))
        self.models_root = Path(os.getenv("MODELS_PATH", "models"))
        self.logs_root = Path(os.getenv("LOGS_PATH", "logs"))
        self.state_root = Path(os.getenv("STATE_PATH", "state"))
        
        # Ensure all base directories exist
        self._ensure_base_directories()
    
    def _ensure_base_directories(self):
        """Create base directories if they don't exist"""
        for base_dir in [self.data_root, self.backup_root, self.models_root, 
                        self.logs_root, self.state_root]:
            base_dir.mkdir(parents=True, exist_ok=True)
    
    def _ensure_path(self, path: Path) -> Path:
        """Ensure directory exists for given path"""
        if path.suffix:  # It's a file
            path.parent.mkdir(parents=True, exist_ok=True)
        else:  # It's a directory
            path.mkdir(parents=True, exist_ok=True)
        return path
    
    # ===== HISTORICAL DATA PATHS =====
    
    def get_historical_data_path(self, 
                                exchange: str, 
                                symbol: str, 
                                timeframe: str,
                                create_dirs: bool = True) -> Path:
        """Get path for historical OHLCV data"""
        path = self.data_root / "historical" / exchange.lower() / timeframe / f"{self._sanitize_symbol(symbol)}.csv"
        return self._ensure_path(path) if create_dirs else path
    
    def get_historical_meta_path(self, 
                                exchange: str, 
                                symbol: str, 
                                timeframe: str,
                                create_dirs: bool = True) -> Path:
        """Get path for historical data metadata"""
        path = self.data_root / "historical" / exchange.lower() / timeframe / f"{self._sanitize_symbol(symbol)}.meta.json"
        return self._ensure_path(path) if create_dirs else path
    
    def get_backup_path(self, exchange: str, symbol: str, timeframe: str = None) -> Path:
        """Get path for backup historical data"""
        if timeframe:
            path = self.backup_root / "historical" / exchange.lower() / timeframe / f"{self._sanitize_symbol(symbol)}.csv"
        else:
            path = self.backup_root / "historical" / exchange.lower() / self._sanitize_symbol(symbol)
        return self._ensure_path(path)
    
    # ===== BRANCH-SCOPED DATA PATHS =====
    
    def get_data_path(self, 
                     branch: str, 
                     mode: str, 
                     dataset_type: str,
                     filename: Optional[str] = None) -> Path:
        """
        Get path for branch/mode-scoped data
        
        Args:
            branch: Strategy branch name (e.g., 'main', 'experimental')
            mode: Trading mode ('paper', 'live', 'backtest')
            dataset_type: Type of dataset ('features', 'indicators', 'signals')
            filename: Optional specific filename
        """
        path = self.data_root / dataset_type / branch / mode
        if filename:
            path = path / filename
        return self._ensure_path(path)
    
    def get_metrics_path(self, 
                        branch: str, 
                        mode: str,
                        filename: Optional[str] = None) -> Path:
        """Get path for performance metrics"""
        path = self.data_root / "metrics" / branch / mode
        if filename:
            path = path / filename
        return self._ensure_path(path)
    
    def get_decisions_path(self, 
                          branch: str, 
                          mode: str,
                          filename: Optional[str] = None) -> Path:
        """Get path for trading decisions and traces"""
        path = self.data_root / "decisions" / branch / mode
        if filename:
            path = path / filename
        return self._ensure_path(path)
    
    # ===== MODEL PATHS =====
    
    def get_model_path(self, 
                      branch: str, 
                      model_type: str = "default",
                      filename: Optional[str] = None) -> Path:
        """Get path for trained models and artifacts"""
        path = self.models_root / branch / model_type
        if filename:
            path = path / filename
        return self._ensure_path(path)
    
    def get_model_checkpoint_path(self, 
                                 branch: str, 
                                 model_type: str = "default",
                                 checkpoint_name: Optional[str] = None) -> Path:
        """Get path for model checkpoints"""
        path = self.models_root / branch / model_type / "checkpoints"
        if checkpoint_name:
            path = path / checkpoint_name
        return self._ensure_path(path)
    
    # ===== LOG PATHS =====
    
    def get_log_path(self, 
                    branch: str, 
                    mode: str, 
                    log_type: str = "main",
                    filename: Optional[str] = None) -> Path:
        """Get path for structured logs"""
        path = self.logs_root / branch / mode / log_type
        if filename:
            path = path / filename
        return self._ensure_path(path)
    
    def get_error_log_path(self, 
                          branch: str,
                          filename: Optional[str] = None) -> Path:
        """Get path for error logs (branch-scoped)"""
        path = self.logs_root / "errors" / branch
        if filename:
            path = path / filename
        return self._ensure_path(path)
    
    def get_structured_log_path(self, filename: str = "structured_logs.jsonl") -> Path:
        """Get path for global structured logs"""
        path = self.logs_root / filename
        return self._ensure_path(path)
    
    # ===== STATE PATHS =====
    
    def get_state_path(self, 
                      branch: str, 
                      mode: str,
                      filename: Optional[str] = None) -> Path:
        """Get path for runtime state"""
        path = self.state_root / branch / mode
        if filename:
            path = path / filename
        return self._ensure_path(path)
    
    def get_position_state_path(self, 
                               branch: str, 
                               mode: str,
                               filename: str = "positions.jsonl") -> Path:
        """Get path for position state tracking"""
        return self.get_state_path(branch, mode, filename)
    
    def get_runtime_state_path(self, 
                              branch: str, 
                              mode: str,
                              filename: str = "runtime.jsonl") -> Path:
        """Get path for runtime state tracking"""
        return self.get_state_path(branch, mode, filename)
    
    # ===== SPECIALIZED PATHS =====
    
    def get_ibkr_cache_path(self, cache_type: str = "historical") -> Path:
        """Get path for IBKR-specific cache data"""
        path = self.data_root / "ibkr" / cache_type
        return self._ensure_path(path)
    
    def get_news_cache_path(self, source: str = "rss") -> Path:
        """Get path for news data cache"""
        path = self.data_root / "news" / source
        return self._ensure_path(path)
    
    def get_telegram_state_path(self, filename: str = "telegram_state.json") -> Path:
        """Get path for Telegram bot state"""
        path = self.state_root / "telegram" / filename
        return self._ensure_path(path)
    
    def get_validation_results_path(self, 
                                   branch: str,
                                   strategy_id: str,
                                   filename: Optional[str] = None) -> Path:
        """Get path for validation results"""
        path = self.data_root / "validation" / branch / strategy_id
        if filename:
            path = path / filename
        return self._ensure_path(path)
    
    # ===== UTILITY METHODS =====
    
    def _sanitize_symbol(self, symbol: str) -> str:
        """Sanitize symbol for filesystem use"""
        return symbol.replace("/", "").replace(":", "").replace("\\", "").upper()
    
    def get_branch_list(self) -> list[str]:
        """Get list of available branches from filesystem"""
        branches = set()
        
        # Scan models directory
        if self.models_root.exists():
            for path in self.models_root.iterdir():
                if path.is_dir():
                    branches.add(path.name)
        
        # Scan data directory
        for data_type in ["metrics", "decisions"]:
            data_path = self.data_root / data_type
            if data_path.exists():
                for path in data_path.iterdir():
                    if path.is_dir():
                        branches.add(path.name)
        
        return sorted(list(branches))
    
    def cleanup_old_data(self, days_old: int = 30):
        """Clean up old temporary data (not implemented - placeholder)"""
        # TODO: Implement cleanup logic for old logs, temp files, etc.
        pass
    
    def get_disk_usage_report(self) -> dict:
        """Get disk usage report for all data directories"""
        report = {}
        
        for name, path in [
            ("data", self.data_root),
            ("backups", self.backup_root),
            ("models", self.models_root),
            ("logs", self.logs_root),
            ("state", self.state_root)
        ]:
            if path.exists():
                total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                file_count = len(list(path.rglob('*')))
                report[name] = {
                    "size_bytes": total_size,
                    "size_mb": round(total_size / 1024 / 1024, 2),
                    "file_count": file_count,
                    "path": str(path)
                }
            else:
                report[name] = {"size_bytes": 0, "size_mb": 0, "file_count": 0, "path": str(path)}
        
        return report


# Global instance
Data_Registry = DataRegistry()


# Convenience functions for backward compatibility
def get_historical_data_path(exchange: str, symbol: str, timeframe: str) -> Path:
    """Backward compatible function"""
    return Data_Registry.get_historical_data_path(exchange, symbol, timeframe)


def get_model_path(branch: str, model_type: str = "default") -> Path:
    """Backward compatible function"""
    return Data_Registry.get_model_path(branch, model_type)


if __name__ == "__main__":
    # Demo usage
    registry = DataRegistry()
    
    print("=== Data Registry Demo ===")
    print(f"Historical BTCUSDT 5m: {registry.get_historical_data_path('bybit', 'BTCUSDT', '5m')}")
    print(f"Main branch metrics: {registry.get_metrics_path('main', 'paper')}")
    print(f"Model path: {registry.get_model_path('main', 'rl_model')}")
    print(f"Log path: {registry.get_log_path('main', 'paper', 'decisions')}")
    print(f"Available branches: {registry.get_branch_list()}")
    print(f"Disk usage: {registry.get_disk_usage_report()}")