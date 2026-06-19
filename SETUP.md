# Setup Guide

## Prerequisites

- Python 3.10+
- A brokerage account (Alpaca recommended for stocks, Robinhood/Webull for options)

## Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd trading-bots

# Install dependencies
pip install alpaca-py yfinance pandas numpy scikit-learn requests
```

## API Keys

### Alpaca (for `turbobot.py` and stock/ETF trading)

1. Sign up at [alpaca.markets](https://alpaca.markets)
2. Go to Paper Trading → API Keys
3. Copy your keys

```bash
# Windows PowerShell
$env:APCA_API_KEY_ID="your_key_here"
$env:APCA_API_SECRET_KEY="your_secret_here"

# Mac/Linux
export APCA_API_KEY_ID="your_key_here"
export APCA_API_SECRET_KEY="your_secret_here"
```

Alternatively, edit `config.py` to hardcode keys (not recommended for shared systems).

### Robinhood/Webull (for 0DTE options bots)

These brokers do not have official Python APIs. For automation:
- **Robinhood:** Use `robin-stocks` library (`pip install robin-stocks`)
- **Webull:** Use `webull` library (`pip install webull`)

The 0DTE strategy bots in `bots/` are designed to work with either broker.

## Running the Bots

### TurboBot (TQQQ on Green SPY Days)

```bash
# Simulation mode (no real trades)
python turbobot.py --sim

# Live mode (requires Alpaca API keys)
python turbobot.py
```

The bot runs continuously. It checks SPY's open at 9:30 AM ET Monday-Friday.
If SPY opens green (>0.3%), it buys TQQQ with a 5% stop loss and holds until close.

### 0DTE Options Bots (Robinhood/Webull)

```bash
cd bots
python trend_continuation.py --sim
python trend_continuation.py --live
```

## Backtesting

```bash
# Mega backtest (all 54 strategies)
python backtests/mega_backtest.py

# Monte Carlo risk simulation
python backtests/monte_carlo.py

# Individual strategy test
cd backtests
python backtest_aggressive.py
```

## File Structure After Setup

```
.
├── config.py          # Edit your API keys here (optional)
├── turbobot.py        # Main bot - ready to run
├── bots/
│   └── *.py           # Additional strategy bots
├── backtests/
│   └── *.py           # Backtest scripts
└── research/
    └── *.py           # Archived experiments
```

## Need Help?

Open a GitHub issue or refer to the detailed [STRATEGIES.md](STRATEGIES.md) for methodology.
