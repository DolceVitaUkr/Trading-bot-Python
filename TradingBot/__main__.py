"""
Run the bot with:  python -m TradingBot
"""
import asyncio
import inspect

def main():
    # Prefer package entry in RunBot/main.py
    try:
        from TradingBot.RunBot.main import main as run
        if inspect.iscoroutinefunction(run):
            return asyncio.run(run())
        else:
            return run()
    except Exception:
        pass

    # Fallback to legacy RunBot.py
    try:
        from TradingBot.RunBot import main as run
        if inspect.iscoroutinefunction(run):
            return asyncio.run(run())
        else:
            return run()
    except Exception:
        pass

    # Fallback to legacy main.py
    try:
        from TradingBot.main import main as run
        if inspect.iscoroutinefunction(run):
            return asyncio.run(run())
        else:
            return run()
    except Exception as e:
        print(f"Could not find a valid entry point. Please check your project structure. Error: {e}")
        return 1

if __name__ == "__main__":
    main()
