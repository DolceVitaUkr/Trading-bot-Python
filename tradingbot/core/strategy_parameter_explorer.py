"""
Strategy Parameter Explorer
Generates and tests diverse trading strategies across comprehensive parameter spaces.
"""
import json
import random
import itertools
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
import numpy as np
from copy import deepcopy

from .loggerconfig import get_logger


@dataclass
class StrategyCandidate:
    """Represents a complete strategy configuration candidate."""
    strategy_id: str
    strategy_type: str
    variant: str
    asset_type: str
    parameters: Dict[str, Any]
    market_specific_params: Dict[str, Any]
    risk_params: Dict[str, Any]
    execution_params: Dict[str, Any]
    fitness_score: Optional[float] = None
    backtest_results: Optional[Dict[str, Any]] = None
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


class StrategyParameterExplorer:
    """
    Comprehensive strategy parameter exploration system.
    Generates diverse trading strategies for testing and optimization.
    """
    
    def __init__(self):
        self.log = get_logger("strategy_explorer")
        
        # Load comprehensive strategy configurations
        self.config_path = Path("tradingbot/config/strategies.json")
        self.comprehensive_config_path = Path("tradingbot/config/comprehensive_strategies.json")
        
        # Load configurations
        self.strategy_config = self._load_config(self.config_path)
        self.comprehensive_config = self._load_config(self.comprehensive_config_path)
        
        # Strategy generation parameters
        self.max_strategies_per_type = 100
        self.parameter_combination_limit = 10000
        
        # Storage for generated strategies
        self.generated_strategies: List[StrategyCandidate] = []
        
        self.log.info("Strategy Parameter Explorer initialized")
        
    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return json.load(f)
            else:
                self.log.warning(f"Config file not found: {config_path}")
                return {}
        except Exception as e:
            self.log.error(f"Error loading config from {config_path}: {e}")
            return {}
    
    def generate_comprehensive_strategies(self, 
                                        asset_types: List[str] = None,
                                        strategy_types: List[str] = None,
                                        max_strategies: int = 1000) -> List[StrategyCandidate]:
        """
        Generate a comprehensive set of strategy candidates for testing.
        
        Args:
            asset_types: List of asset types to generate for ['crypto', 'forex']
            strategy_types: List of strategy types to include
            max_strategies: Maximum number of strategies to generate
            
        Returns:
            List of strategy candidates ready for testing
        """
        if asset_types is None:
            asset_types = ['crypto', 'crypto_futures', 'forex', 'forex_options']
        
        if strategy_types is None:
            strategy_types = list(self.strategy_config.get('active_strategy_types', {}).keys())
        
        self.log.info(f"Generating comprehensive strategies for {asset_types} across {strategy_types}")
        
        all_strategies = []
        
        for asset_type in asset_types:
            for strategy_type in strategy_types:
                strategies = self._generate_strategies_for_type(
                    asset_type=asset_type,
                    strategy_type=strategy_type,
                    max_count=min(max_strategies // (len(asset_types) * len(strategy_types)), 
                                self.max_strategies_per_type)
                )
                all_strategies.extend(strategies)
        
        # Shuffle and limit total strategies
        random.shuffle(all_strategies)
        self.generated_strategies = all_strategies[:max_strategies]
        
        self.log.info(f"Generated {len(self.generated_strategies)} strategy candidates")
        return self.generated_strategies
    
    def _generate_strategies_for_type(self, 
                                    asset_type: str, 
                                    strategy_type: str, 
                                    max_count: int) -> List[StrategyCandidate]:
        """Generate strategy candidates for a specific type and asset."""
        strategies = []
        
        # Get strategy configuration
        strategy_config = self.strategy_config.get('active_strategy_types', {}).get(strategy_type, {})
        if not strategy_config:
            self.log.warning(f"No configuration found for strategy type: {strategy_type}")
            return strategies
        
        variants = strategy_config.get('variants', [strategy_type])  # Fallback to strategy type
        base_parameters = strategy_config.get('parameters', {})
        
        # Generate parameter combinations
        param_combinations = self._generate_parameter_combinations(base_parameters, max_count)
        
        for i, param_combo in enumerate(param_combinations[:max_count]):
            for variant in variants:
                if len(strategies) >= max_count:
                    break
                
                # Create strategy candidate
                strategy = self._create_strategy_candidate(
                    asset_type=asset_type,
                    strategy_type=strategy_type,
                    variant=variant,
                    parameters=param_combo
                )
                
                strategies.append(strategy)
        
        self.log.debug(f"Generated {len(strategies)} candidates for {asset_type} {strategy_type}")
        return strategies
    
    def _generate_parameter_combinations(self, 
                                       base_parameters: Dict[str, Any], 
                                       max_combinations: int) -> List[Dict[str, Any]]:
        """Generate parameter combinations using various sampling methods."""
        param_combinations = []
        
        # Extract parameter ranges
        param_ranges = {}
        for param_name, param_values in base_parameters.items():
            if isinstance(param_values, list):
                param_ranges[param_name] = param_values
            else:
                param_ranges[param_name] = [param_values]  # Single value
        
        if not param_ranges:
            return [{}]
        
        # Calculate total possible combinations
        total_combinations = 1
        for values in param_ranges.values():
            total_combinations *= len(values)
        
        if total_combinations <= max_combinations:
            # Use exhaustive grid search
            param_combinations = self._exhaustive_grid_search(param_ranges)
        else:
            # Use sampling methods
            param_combinations = self._sample_parameter_space(param_ranges, max_combinations)
        
        return param_combinations
    
    def _exhaustive_grid_search(self, param_ranges: Dict[str, List]) -> List[Dict[str, Any]]:
        """Generate all possible parameter combinations."""
        if not param_ranges:
            return [{}]
        
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        combinations = []
        for combination in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combination))
            combinations.append(param_dict)
        
        return combinations
    
    def _sample_parameter_space(self, param_ranges: Dict[str, List], max_samples: int) -> List[Dict[str, Any]]:
        """Sample parameter space using multiple methods."""
        combinations = []
        
        # Method 1: Random sampling (50% of samples)
        random_samples = max_samples // 2
        for _ in range(random_samples):
            param_dict = {}
            for param_name, param_values in param_ranges.items():
                param_dict[param_name] = random.choice(param_values)
            combinations.append(param_dict)
        
        # Method 2: Latin Hypercube-like sampling (25% of samples)
        lhs_samples = max_samples // 4
        combinations.extend(self._latin_hypercube_sampling(param_ranges, lhs_samples))
        
        # Method 3: Boundary exploration (25% of samples)
        boundary_samples = max_samples - len(combinations)
        combinations.extend(self._boundary_exploration(param_ranges, boundary_samples))
        
        return combinations[:max_samples]
    
    def _latin_hypercube_sampling(self, param_ranges: Dict[str, List], n_samples: int) -> List[Dict[str, Any]]:
        """Simplified Latin Hypercube sampling for parameter exploration."""
        if n_samples <= 0:
            return []
        
        combinations = []
        param_names = list(param_ranges.keys())
        
        for _ in range(n_samples):
            param_dict = {}
            for param_name in param_names:
                param_values = param_ranges[param_name]
                # Divide range into segments and sample from each
                segment_size = len(param_values) / n_samples
                segment_idx = int(random.uniform(0, 1) * n_samples)
                value_idx = int(segment_idx * segment_size) % len(param_values)
                param_dict[param_name] = param_values[value_idx]
            combinations.append(param_dict)
        
        return combinations
    
    def _boundary_exploration(self, param_ranges: Dict[str, List], n_samples: int) -> List[Dict[str, Any]]:
        """Explore parameter space boundaries."""
        if n_samples <= 0:
            return []
        
        combinations = []
        param_names = list(param_ranges.keys())
        
        # Generate boundary combinations
        for _ in range(n_samples):
            param_dict = {}
            for param_name in param_names:
                param_values = param_ranges[param_name]
                # Choose from extremes more often
                if random.random() < 0.7:  # 70% chance to pick extreme values
                    param_dict[param_name] = random.choice([param_values[0], param_values[-1]])
                else:
                    param_dict[param_name] = random.choice(param_values)
            combinations.append(param_dict)
        
        return combinations
    
    def _create_strategy_candidate(self, 
                                 asset_type: str,
                                 strategy_type: str, 
                                 variant: str,
                                 parameters: Dict[str, Any]) -> StrategyCandidate:
        """Create a complete strategy candidate with all parameters."""
        
        # Generate unique strategy ID
        strategy_id = f"{asset_type}_{strategy_type}_{variant}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
        
        # Get market-specific parameters
        market_params = self._get_market_specific_parameters(asset_type)
        
        # Get risk management parameters
        risk_params = self._get_risk_management_parameters(asset_type)
        
        # Get execution parameters
        execution_params = self._get_execution_parameters()
        
        # Create comprehensive strategy candidate
        strategy = StrategyCandidate(
            strategy_id=strategy_id,
            strategy_type=strategy_type,
            variant=variant,
            asset_type=asset_type,
            parameters=parameters,
            market_specific_params=market_params,
            risk_params=risk_params,
            execution_params=execution_params
        )
        
        return strategy
    
    def _get_market_specific_parameters(self, asset_type: str) -> Dict[str, Any]:
        """Get market-specific parameters for asset type."""
        market_adaptations = self.strategy_config.get('market_specific_adaptations', {})
        asset_config = market_adaptations.get(asset_type, {})
        
        params = {}
        for param_name, param_values in asset_config.items():
            if isinstance(param_values, list):
                params[param_name] = random.choice(param_values)
            else:
                params[param_name] = param_values
        
        return params
    
    def _get_risk_management_parameters(self, asset_type: str = "spot") -> Dict[str, Any]:
        """Generate risk management parameters based on asset type."""
        risk_config = self.strategy_config.get('risk_management_parameters', {})
        
        # Determine risk category based on asset type
        if 'futures' in asset_type.lower():
            risk_category = 'futures_trading'
        elif 'options' in asset_type.lower():
            risk_category = 'options_trading'
        else:
            risk_category = 'spot_trading'
        
        # Get risk-specific configuration
        risk_specific_config = risk_config.get(risk_category, {})
        portfolio_config = risk_config.get('portfolio_constraints', {})
        
        params = {}
        params['risk_category'] = risk_category
        
        # Asset-specific risk parameters
        if risk_category == 'futures_trading':
            # Leverage management
            leverage_config = risk_specific_config.get('leverage_management', {})
            params['max_leverage'] = random.choice(leverage_config.get('max_leverage', [10]))
            params['dynamic_leverage'] = random.choice(leverage_config.get('dynamic_leverage', [False]))
            
            # Margin management
            margin_config = risk_specific_config.get('margin_management', {})
            params['initial_margin_buffer'] = random.choice(margin_config.get('initial_margin_buffer', [0.2]))
            params['liquidation_threshold'] = random.choice(margin_config.get('liquidation_avoidance_threshold', [0.7]))
            
            # Funding rate management
            funding_config = risk_specific_config.get('funding_rate_management', {})
            params['funding_rate_threshold'] = random.choice(funding_config.get('funding_rate_thresholds', [0.1]))
            params['funding_arbitrage'] = random.choice(funding_config.get('funding_arbitrage', [False]))
            
        elif risk_category == 'options_trading':
            # Greeks management
            greeks_config = risk_specific_config.get('greeks_management', {})
            params['delta_limit'] = random.choice(greeks_config.get('delta_limits', [0.3]))
            params['gamma_limit'] = random.choice(greeks_config.get('gamma_limits', [0.1]))
            params['theta_target'] = random.choice(greeks_config.get('theta_targets', [0]))
            params['vega_limit'] = random.choice(greeks_config.get('vega_limits', [500]))
            
            # Volatility management
            vol_config = risk_specific_config.get('volatility_management', {})
            params['implied_vol_range'] = random.choice(vol_config.get('implied_vol_range', [25]))
            params['vol_skew_limit'] = random.choice(vol_config.get('vol_skew_limits', [0.2]))
            
            # Time decay management
            time_config = risk_specific_config.get('time_decay_management', {})
            params['theta_strategy'] = random.choice(time_config.get('theta_decay_strategies', ['theta_neutral']))
            params['dte_limit'] = random.choice(time_config.get('time_to_expiration_limits', [30]))
        
        # Position sizing
        sizing_config = risk_specific_config.get('position_sizing', {})
        params['position_sizing_method'] = random.choice(sizing_config.get('methods', ['fixed_percentage']))
        params['risk_percentage'] = random.choice(sizing_config.get('risk_percentages', [1.0]))
        
        # Stop loss
        sl_config = risk_specific_config.get('stop_loss_strategies', {})
        params['stop_loss_type'] = random.choice(sl_config.get('types', ['fixed_percentage']))
        
        if params['stop_loss_type'] == 'fixed_percentage':
            params['stop_loss_value'] = random.choice(sl_config.get('fixed_percentages', [2.0]))
        elif params['stop_loss_type'] == 'atr_based':
            params['stop_loss_value'] = random.choice(sl_config.get('atr_multipliers', [2.0]))
        
        # Take profit (for futures and spot)
        if risk_category != 'options_trading':
            tp_config = risk_specific_config.get('take_profit_strategies', {})
            params['take_profit_type'] = random.choice(tp_config.get('types', ['fixed_ratio']))
            
            if params['take_profit_type'] == 'fixed_ratio':
                params['take_profit_value'] = random.choice(tp_config.get('risk_reward_ratios', [2.0]))
            elif params['take_profit_type'] == 'atr_based':
                params['take_profit_value'] = random.choice(tp_config.get('atr_multipliers', [3.0]))
        
        # Portfolio constraints
        params['max_concurrent_positions'] = random.choice(portfolio_config.get('max_concurrent_positions', [3]))
        params['max_daily_trades'] = random.choice(portfolio_config.get('max_daily_trades', [10]))
        
        return params
    
    def _get_execution_parameters(self) -> Dict[str, Any]:
        """Generate execution parameters."""
        execution_config = self.strategy_config.get('execution_parameters', {})
        
        params = {}
        
        # Timeframes
        timeframe_config = execution_config.get('timeframes', {})
        params['primary_timeframe'] = random.choice(timeframe_config.get('primary_analysis', ['15m']))
        params['entry_timeframe'] = random.choice(timeframe_config.get('entry_timeframes', ['5m']))
        
        # Order execution
        order_config = execution_config.get('order_execution', {})
        params['entry_order_type'] = random.choice(order_config.get('entry_order_types', ['market']))
        params['exit_order_type'] = random.choice(order_config.get('exit_order_types', ['market']))
        params['slippage_tolerance'] = random.choice(order_config.get('slippage_tolerance', [0.005]))
        
        # Market condition filters
        condition_config = execution_config.get('market_condition_filters', {})
        params['volatility_filter'] = random.choice(condition_config.get('volatility_filters', ['adaptive']))
        params['trend_filter'] = random.choice(condition_config.get('trend_filters', ['adaptive']))
        params['liquidity_filter'] = random.choice(condition_config.get('liquidity_filters', ['medium_liquidity_ok']))
        
        return params
    
    def get_strategy_diversity_report(self) -> Dict[str, Any]:
        """Generate a report on strategy diversity and coverage."""
        if not self.generated_strategies:
            return {"error": "No strategies generated yet"}
        
        # Analyze strategy distribution
        strategy_types = {}
        asset_types = {}
        variants = {}
        
        for strategy in self.generated_strategies:
            # Count strategy types
            if strategy.strategy_type not in strategy_types:
                strategy_types[strategy.strategy_type] = 0
            strategy_types[strategy.strategy_type] += 1
            
            # Count asset types
            if strategy.asset_type not in asset_types:
                asset_types[strategy.asset_type] = 0
            asset_types[strategy.asset_type] += 1
            
            # Count variants
            variant_key = f"{strategy.strategy_type}_{strategy.variant}"
            if variant_key not in variants:
                variants[variant_key] = 0
            variants[variant_key] += 1
        
        # Calculate parameter diversity
        parameter_diversity = self._calculate_parameter_diversity()
        
        report = {
            "total_strategies": len(self.generated_strategies),
            "strategy_type_distribution": strategy_types,
            "asset_type_distribution": asset_types,
            "variant_distribution": variants,
            "parameter_diversity": parameter_diversity,
            "unique_combinations": len(set(str(s.parameters) for s in self.generated_strategies)),
            "generation_timestamp": datetime.now().isoformat()
        }
        
        return report
    
    def _calculate_parameter_diversity(self) -> Dict[str, Any]:
        """Calculate diversity metrics for generated parameters."""
        if not self.generated_strategies:
            return {}
        
        # Collect all parameter values
        all_params = {}
        for strategy in self.generated_strategies:
            for param_name, param_value in strategy.parameters.items():
                if param_name not in all_params:
                    all_params[param_name] = []
                all_params[param_name].append(param_value)
        
        # Calculate diversity for each parameter
        diversity_metrics = {}
        for param_name, values in all_params.items():
            unique_values = len(set(str(v) for v in values))
            total_values = len(values)
            diversity_ratio = unique_values / total_values if total_values > 0 else 0
            
            diversity_metrics[param_name] = {
                "unique_values": unique_values,
                "total_occurrences": total_values,
                "diversity_ratio": round(diversity_ratio, 3),
                "sample_values": list(set(str(v) for v in values))[:10]  # Show first 10 unique values
            }
        
        return diversity_metrics
    
    def export_strategies_for_testing(self, output_path: str = None) -> str:
        """Export generated strategies to JSON file for testing."""
        if not self.generated_strategies:
            raise ValueError("No strategies generated yet. Call generate_comprehensive_strategies() first.")
        
        if output_path is None:
            output_path = f"tradingbot/state/generated_strategies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert strategies to dictionaries
        strategies_data = []
        for strategy in self.generated_strategies:
            strategy_dict = {
                "strategy_id": strategy.strategy_id,
                "strategy_type": strategy.strategy_type,
                "variant": strategy.variant,
                "asset_type": strategy.asset_type,
                "parameters": strategy.parameters,
                "market_specific_params": strategy.market_specific_params,
                "risk_params": strategy.risk_params,
                "execution_params": strategy.execution_params,
                "created_at": strategy.created_at
            }
            strategies_data.append(strategy_dict)
        
        # Save to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        export_data = {
            "metadata": {
                "total_strategies": len(strategies_data),
                "export_timestamp": datetime.now().isoformat(),
                "explorer_version": "2.0"
            },
            "diversity_report": self.get_strategy_diversity_report(),
            "strategies": strategies_data
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.log.info(f"Exported {len(strategies_data)} strategies to {output_file}")
        return str(output_file)
    
    def generate_exploration_summary(self) -> str:
        """Generate a human-readable summary of strategy exploration."""
        if not self.generated_strategies:
            return "No strategies generated yet."
        
        diversity_report = self.get_strategy_diversity_report()
        
        summary = f"""
🚀 COMPREHENSIVE STRATEGY EXPLORATION SUMMARY 🚀

📊 GENERATION OVERVIEW:
• Total Strategies Generated: {diversity_report['total_strategies']}
• Unique Parameter Combinations: {diversity_report['unique_combinations']}
• Generation Time: {diversity_report['generation_timestamp']}

📈 STRATEGY TYPE DISTRIBUTION:
"""
        
        for strategy_type, count in diversity_report['strategy_type_distribution'].items():
            percentage = (count / diversity_report['total_strategies']) * 100
            summary += f"• {strategy_type.replace('_', ' ').title()}: {count} strategies ({percentage:.1f}%)\n"
        
        summary += f"""
🌍 ASSET TYPE COVERAGE:
"""
        
        for asset_type, count in diversity_report['asset_type_distribution'].items():
            percentage = (count / diversity_report['total_strategies']) * 100
            summary += f"• {asset_type.upper()}: {count} strategies ({percentage:.1f}%)\n"
        
        summary += f"""
🎯 TOP STRATEGY VARIANTS:
"""
        
        # Show top 10 variants
        sorted_variants = sorted(diversity_report['variant_distribution'].items(), key=lambda x: x[1], reverse=True)
        for variant, count in sorted_variants[:10]:
            summary += f"• {variant.replace('_', ' ').title()}: {count} strategies\n"
        
        summary += f"""
🔬 PARAMETER EXPLORATION DEPTH:
• The bot will now explore {diversity_report['unique_combinations']} unique parameter combinations
• This includes variations in:
  - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
  - Moving averages (EMA, SMA, WMA, Hull MA)
  - Risk management (Stop losses, Take profits, Position sizing)
  - Market timing (Multiple timeframes, session filters)
  - Entry/Exit strategies (Breakouts, Mean reversion, Momentum)
  - Volatility strategies (Squeeze, Expansion, ATR-based)
  - Pattern recognition (Candlestick, Chart patterns)
  - And much more!

🎲 EXPLORATION METHODOLOGY:
• Grid Search: Exhaustive testing of key combinations
• Random Sampling: Diverse parameter exploration  
• Boundary Testing: Extreme parameter values
• Latin Hypercube: Efficient space coverage

This comprehensive exploration will help discover the most profitable trading strategies
across different market conditions for both crypto and forex markets! 🎯
"""
        
        return summary


# Convenience function for easy usage
def generate_diverse_strategies(asset_types: List[str] = None, 
                              max_strategies: int = 1000) -> Tuple[List[StrategyCandidate], str, str]:
    """
    Convenience function to generate diverse strategies and return summary.
    
    Returns:
        Tuple of (strategies_list, summary_text, export_path)
    """
    explorer = StrategyParameterExplorer()
    strategies = explorer.generate_comprehensive_strategies(
        asset_types=asset_types or ['crypto', 'crypto_futures', 'forex', 'forex_options'],
        max_strategies=max_strategies
    )
    
    summary = explorer.generate_exploration_summary()
    
    # Export strategies
    export_path = explorer.export_strategies_for_testing()
    
    return strategies, summary, export_path


__all__ = [
    "StrategyParameterExplorer", 
    "StrategyCandidate", 
    "generate_diverse_strategies"
]