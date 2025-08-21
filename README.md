README — Naming & Structure (updated inserts)
File & Naming Guidelines

Files (modules): all lowercase, no underscores (e.g., runtimecontroller.py, validationmanager.py).

Functions: all lowercase, no underscores (e.g., enablelive, disableslive, routeliveorder).

Classes: CamelCase (e.g., RuntimeController, ValidationManager).

Every source file starts with:

# file: core/datamanager.py

Project Structure (module filenames updated)
tradingbot/
  core/
    configmanager.py
    runtimecontroller.py
    datamanager.py
    featurestore.py
    indicators.py
    pairmanager.py
    riskmanager.py
    tradeexecutor.py
    portfoliomanager.py
    validationmanager.py
    optimizer.py
    notifier.py
    errorhandler.py
    driftmonitor.py
  brokers/
    exchangebybit.py
    exchangeibkr.py
  learning/
    statefeaturizer.py
    trainmlmodel.py
    trainrlmodel.py
    saveaiupdate.py
  ui/
    app.py
    routes/
      diff.py
      validation.py
  config/
    config.json
    assets.json
    strategies.json
  logs/
  state/
  tests/
