# Bots Collection

This directory contains trading bots for various strategies. See [STRATEGIES.md](../STRATEGIES.md) for full ranking.

## Available Bots

| Bot | Strategy | WR | $/hr | Capital | Needs |
|-----|----------|----|------|---------|-------|
| `turbobot.py` | TQQQ on green SPY days | 54.7%* | $0.001 | $200 | **Alpaca** |
| `trend_continuation.py` | 0DTE + trend filter | **65.9%** | $0.75 | $270 | Options broker |
| `ensemble_bot.py` | Multi-signal voting | 53.1% | $0.70 | $290 | Options broker |
| `pattern_bot.py` | Candlestick patterns | **94.3%** | $0.20 | $1,000 | Options broker |

*\*86% per-trade opening print WR, 54.7% EOD WR.*

## Which One Should I Run?

1. **$200 on Alpaca** → `python ../turbobot.py --sim` (TQQQ on green SPY days)
2. **$200+ on Robinhood/Webull** → `python trend_continuation.py` (0DTE + trend filter)
3. **$1,000+ safe growth** → `python pattern_bot.py` (94.3% WR pattern recognition)
4. **Need steady income** → `python ensemble_bot.py` (multi-signal voting)
