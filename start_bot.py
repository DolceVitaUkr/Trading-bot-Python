#!/usr/bin/env python3
"""
Trading Bot Launcher Script with Multi-Asset Paper → Pre-Validator CLI
"""
import sys
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Ensure we're in the right directory
project_root = Path(__file__).parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

def run_validation_pipeline(asset: str, strategy: str, symbols: List[str], 
                           start: str, end: str, output_dir: str = None) -> Dict[str, Any]:
    """
    Run the complete Multi-Asset Paper → Pre-Validator pipeline.
    
    Args:
        asset: Asset type (crypto_spot, crypto_futures, forex, forex_options)
        strategy: Strategy name/identifier  
        symbols: List of trading symbols
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        output_dir: Output directory (defaults to results/{strategy}_{asset})
        
    Returns:
        Dictionary containing complete validation results
    """
    try:
        # Import required modules
        from tradingbot.core.paper_trader import PaperTrader
        from tradingbot.core.strategy_manager import StrategyManager
        from tradingbot.core.validation_manager import ValidationManager
        from tradingbot.core.telegrambot import TelegramNotifier
        
        print(f"\n🚀 Starting Multi-Asset Validation Pipeline")
        print(f"Strategy: {strategy}")
        print(f"Asset: {asset}")
        print(f"Symbols: {', '.join(symbols)}")
        print(f"Period: {start} to {end}")
        print("=" * 60)
        
        # Set up output directory
        if output_dir is None:
            output_dir = f"results/{strategy}_{asset}_{start}_{end}"
        result_path = Path(output_dir)
        result_path.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Output directory: {result_path}")
        
        # Initialize managers
        paper_trader = PaperTrader()
        strategy_manager = StrategyManager()
        validation_manager = ValidationManager()
        telegram_notifier = TelegramNotifier()
        
        # Step 1: Run Paper Trades
        print(f"\n📈 Step 1: Running paper trades simulation...")
        paper_results = paper_trader.Run_Paper_Trades(
            asset=asset,
            strategy=strategy, 
            symbols=symbols,
            start=start,
            end=end,
            result_path=str(result_path)
        )
        print(f"✅ Paper trades completed. Total trades: {paper_results.get('total_trades', 0)}")
        
        # Step 2: Compute KPIs
        print(f"\n📊 Step 2: Computing performance KPIs...")
        kpis = strategy_manager.Compute_KPIs(asset=asset, result_path=str(result_path))
        sharpe = kpis.get('sharpe_ratio', 0)
        max_dd = kpis.get('max_drawdown', 0)
        win_rate = kpis.get('win_rate', 0)
        print(f"✅ KPIs computed. Sharpe: {sharpe:.2f}, Max DD: {max_dd:.1%}, Win Rate: {win_rate:.1%}")
        
        # Step 3: Generate Baselines
        print(f"\n🎯 Step 3: Generating baseline comparisons...")
        baselines = strategy_manager.Generate_Baselines(
            asset=asset,
            symbols=symbols,
            start=start,
            end=end,
            result_path=str(result_path)
        )
        print(f"✅ Baselines generated. Count: {len(baselines)}")
        
        # Step 4: Robustness Checks
        print(f"\n🔬 Step 4: Running robustness tests...")
        robustness = validation_manager.Robustness_Checks(
            asset=asset,
            strategy_result_path=str(result_path),
            baseline_results=baselines
        )
        robustness_pass = robustness.get('overall_assessment', {}).get('overall_pass', False)
        print(f"✅ Robustness tests {'PASSED' if robustness_pass else 'FAILED'}")
        
        # Step 5: Risk Compliance Checks
        print(f"\n⚖️ Step 5: Checking risk compliance...")
        compliance = validation_manager.Risk_Compliance_Checks(
            asset=asset,
            strategy_result_path=str(result_path)
        )
        compliance_pass = compliance.get('overall_compliance', {}).get('fully_compliant', False)
        print(f"✅ Compliance checks {'PASSED' if compliance_pass else 'FAILED'}")
        
        # Step 6: Prepare Validator Package
        print(f"\n📦 Step 6: Preparing validation package...")
        validation_package = strategy_manager.Prepare_Validator_Package(
            asset=asset,
            strategy=strategy,
            kpis=kpis,
            baselines=baselines,
            robustness=robustness,
            compliance=compliance,
            result_path=str(result_path)
        )
        
        final_approved = validation_package.get('final_verdict', {}).get('approved', False)
        final_score = validation_package.get('validation_scores', {}).get('final_score', 0)
        print(f"✅ Validation package prepared. Status: {'APPROVED' if final_approved else 'REJECTED'}")
        print(f"📋 Final Score: {final_score:.1%}")
        
        # Step 7: Send Telegram Notification
        print(f"\n📱 Step 7: Sending Telegram notification...")
        telegram_result = telegram_notifier.Notify_Telegram_Update(
            asset=asset,
            strategy=strategy,
            validation_result=validation_package,
            result_path=str(result_path)
        )
        telegram_sent = telegram_result.get('sent', False)
        print(f"✅ Telegram notification {'sent' if telegram_sent else 'failed'}")
        
        # Final Summary
        print(f"\n🎉 PIPELINE COMPLETED")
        print("=" * 60)
        print(f"Strategy: {strategy} ({asset})")
        print(f"Final Status: {'✅ APPROVED' if final_approved else '❌ REJECTED'}")
        print(f"Overall Score: {final_score:.1%}")
        print(f"Key Metrics:")
        print(f"  • Sharpe Ratio: {sharpe:.2f}")
        print(f"  • Max Drawdown: {max_dd:.1%}")  
        print(f"  • Win Rate: {win_rate:.1%}")
        print(f"  • Profit Factor: {kpis.get('profit_factor', 0):.2f}")
        print(f"Files saved to: {result_path}")
        print("=" * 60)
        
        return validation_package
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        raise


def start_dashboard():
    """Start the web dashboard."""
    print("Starting Trading Bot Dashboard...")
    print(f"Working directory: {os.getcwd()}")

    try:
        from tradingbot.ui.app import app
        import uvicorn
        
        print("Launching web dashboard...")
        print("Dashboard will be available at:")
        print("   - Local: http://127.0.0.1:8000")
        print("   - Network: http://0.0.0.0:8000")
        print("\nPress Ctrl+C to stop the bot")
        print("=" * 50)
        
        # Start the server
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8000, 
            log_level="info",
            reload=False  # Set to True for development
        )
        
    except ImportError as e:
        print(f"Import Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're in the Trading-bot-Python directory")
        print("2. Install required dependencies: pip install -r requirements.txt")
        print("3. Check that all files are present")
        
    except KeyboardInterrupt:
        print("\n\nTrading Bot stopped by user")
        print("All data has been saved")
        
    except Exception as e:
        print(f"Error starting bot: {e}")
        print("\nCheck the logs above for more details")


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Trading Bot with Multi-Asset Paper → Pre-Validator Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start web dashboard (default)
  python start_bot.py

  # Run validation pipeline for crypto spot strategy
  python start_bot.py validate --asset crypto_spot --strategy my_strategy --symbols BTCUSDT,ETHUSDT --start 2024-01-01 --end 2024-06-30

  # Run validation with custom output directory
  python start_bot.py validate --asset forex --strategy eur_usd_scalper --symbols EURUSD --start 2024-01-01 --end 2024-12-31 --output results/eur_scalper_2024
        """
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Dashboard command (default)
    dashboard_parser = subparsers.add_parser('dashboard', help='Start web dashboard (default)')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Run validation pipeline')
    validate_parser.add_argument('--asset', required=True, 
                               choices=['crypto_spot', 'crypto_futures', 'forex', 'forex_options'],
                               help='Asset type to validate')
    validate_parser.add_argument('--strategy', required=True,
                               help='Strategy name/identifier')
    validate_parser.add_argument('--symbols', required=True,
                               help='Comma-separated list of symbols (e.g., BTCUSDT,ETHUSDT)')
    validate_parser.add_argument('--start', required=True,
                               help='Start date (YYYY-MM-DD)')
    validate_parser.add_argument('--end', required=True,
                               help='End date (YYYY-MM-DD)')
    validate_parser.add_argument('--output', default=None,
                               help='Output directory (optional)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle commands
    if args.command == 'validate':
        # Parse symbols
        symbols = [s.strip() for s in args.symbols.split(',')]
        
        # Run validation pipeline
        try:
            results = run_validation_pipeline(
                asset=args.asset,
                strategy=args.strategy,
                symbols=symbols,
                start=args.start,
                end=args.end,
                output_dir=args.output
            )
            
            # Exit with success/failure code based on approval
            approved = results.get('final_verdict', {}).get('approved', False)
            sys.exit(0 if approved else 1)
            
        except Exception as e:
            print(f"Validation pipeline failed: {e}")
            sys.exit(1)
    
    else:
        # Default: start dashboard
        start_dashboard()


if __name__ == "__main__":
    main()