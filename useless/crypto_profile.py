"""Crypto trading profile for maximum profit per hour.

Activated by setting trading_config.crypto_mode = True.
Switches exit params, symbol universe, and ML training to crypto-optimized settings.
"""

# Hard stop: 1.0% loss (vs 0.4% for stocks)
HARD_SL_PCT = 1.0

# Trail activates at 0.5% gain (vs 0.3% for stocks)
TRAIL_ACT_PCT = 0.5

# Trail distance 0.4% (vs 0.3% for stocks)
TRAIL_DIST_PCT = 0.4

# ML training: use shorter, more recent data for crypto
# Crypto regime changes faster than stocks; older data hurts accuracy
ML_DATA_PERIOD = "2mo"

# --- Symbol Lists (yfinance format = "-USD") ---
# The engine converts to Alpaca format ("/USD") for order execution.

# Full crypto universe (yfinance format for scanner compatibility)
SYMBOLS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD",
    "ADA-USD", "AVAX-USD", "DOT-USD", "LINK-USD", "LTC-USD",
    "BCH-USD", "UNI-USD", "ATOM-USD", "NEAR-USD", "APT-USD",
    "SUI-USD", "PEPE-USD", "WIF-USD", "INJ-USD", "TIA-USD",
    "SEI-USD", "FET-USD", "ARB-USD", "OP-USD", "BONK-USD",
    "FLOKI-USD", "PENDLE-USD", "JTO-USD", "WLD-USD", "ENA-USD",
    "ETHFI-USD", "ALT-USD", "OMNI-USD", "STRK-USD", "AKRO-USD",
]

# Hot symbols for fast scan (most volatile / momentum-prone pairs)
HOT_SYMBOLS = [
    "SOL-USD", "DOGE-USD", "PEPE-USD", "WIF-USD", "BONK-USD",
    "AVAX-USD", "SUI-USD", "APT-USD", "INJ-USD", "SEI-USD",
    "NEAR-USD", "TIA-USD", "ARB-USD", "OP-USD", "FET-USD",
    "ETH-USD", "BTC-USD", "XRP-USD", "ADA-USD", "DOT-USD",
    "FLOKI-USD", "PENDLE-USD", "JTO-USD", "WLD-USD", "ENA-USD",
]

# Alpaca-formatted subset for order execution (only pairs Alpaca supports)
ALPACA_SYMBOLS = [s.replace("-USD", "/USD") for s in SYMBOLS]
