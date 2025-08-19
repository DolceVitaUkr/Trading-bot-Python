"""
Run the bot with:  python -m trading_bot
"""
import asyncio
import inspect

def main():
    # Prefer package entry in run_bot/main.py
    try:
        from trading_bot.run_bot.main import main as run
        if inspect.iscoroutinefunction(run):
            return asyncio.run(run())
        else:
            return run()
    except Exception:
        pass

    # Fallback to legacy run_bot.py
    try:
        from trading_bot.run_bot import main as run
        if inspect.iscoroutinefunction(run):
            return asyncio.run(run())
        else:
            return run()
    except Exception:
        pass

    # Fallback to legacy main.py
    try:
        from trading_bot.main import main as run
        if inspect.iscoroutinefunction(run):
            return asyncio.run(run())
        else:
            return run()
    except Exception as e:
        print(f"Could not find a valid entry point. Please check your project structure. Error: {e}")
        return 1

if __name__ == "__main__":
    main()
