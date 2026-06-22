#!/bin/bash
# =============================================================================
# $100 → $1,000,000,000 BOT — ALL-IN MOMENTUM 24/7
# =============================================================================
# Strategy:
#   - Scans symbols (stocks or crypto) via yfinance every 5s
#   - ALL-IN on single best momentum pick at any moment
#   - Trail-only exit (no TP cap) — lets winners run
#   - No risk limits, no diversification, pure aggression
#   - Fractional shares, auto-reconnect, compounding tracker
#
# SWITCHING TO CRYPTO MODE:
#   Add this line AFTER "trading_config.paper_trading = True":
#     trading_config.crypto_mode = True
#   Then restart. Uses 1% SL / 0.5% trail / 35 crypto pairs.
# =============================================================================
cd "$(dirname "$0")"

CERT_FILE=$(python3 -c "import certifi; print(certifi.where())" 2>/dev/null)
export SSL_CERT_FILE="$CERT_FILE"
export REQUESTS_CA_BUNDLE="$CERT_FILE"
export ALPACA_API_KEY="PK2P65MFD5WDNHY7D6R276EAVE"
export ALPACA_SECRET_KEY="Fd3QBzSVfQfngoRn9ZQxHqTtjSthzHtY7ezPg3Yogm6m"

echo ""
echo "🔥 $100 → $1,000,000,000 BOT 🔥"
echo "========================================"
echo "Mode:    ALL-IN MOMENTUM 24/7"
echo "Scanner: yfinance every 5s + hot-universe fast scan"
echo "Exits:   Trail-only (no TP) — winners run"
echo "Risk:    NONE — 100% allocation, no limits"
echo "Hours:   24/7 (crypto + extended hours equities)"
echo "Log:     trading_engine.log"
echo "========================================"
echo ""

python3 -c "
import os, sys
sys.path.insert(0, '.')
os.environ['SSL_CERT_FILE'] = '$CERT_FILE'
os.environ['REQUESTS_CA_BUNDLE'] = '$CERT_FILE'

from config import alpaca_config, trading_config, model_config, sentiment_config, system_config
from trading_engine import MultiModalTradingEngine

trading_config.dry_run = False
trading_config.paper_trading = True
alpaca_config.api_key = 'PK2P65MFD5WDNHY7D6R276EAVE'
alpaca_config.secret_key = 'Fd3QBzSVfQfngoRn9ZQxHqTtjSthzHtY7ezPg3Yogm6m'

# Uncomment to switch to crypto-only mode (1% SL / 0.5% trail / 35 pairs):
# trading_config.crypto_mode = True

engine = MultiModalTradingEngine(alpaca_config, trading_config, model_config, sentiment_config, system_config)
import asyncio
asyncio.run(engine.run())
" 2>&1 | tee -a trading_engine.log