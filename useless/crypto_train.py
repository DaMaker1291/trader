"""Train ML model on crypto data for crypto_profile."""
import os, asyncio

os.environ["ALPACA_API_KEY"] = "PK2P65MFD5WDNHY7D6R276EAVE"
os.environ["ALPACA_SECRET_KEY"] = "Fd3QBzSVfQfngoRn9ZQxHqTtjSthzHtY7ezPg3Yogm6m"

from config import alpaca_config, trading_config, model_config, sentiment_config, system_config
from trading_engine import MultiModalTradingEngine

trading_config.dry_run = True
trading_config.paper_trading = True
trading_config.crypto_mode = True

engine = MultiModalTradingEngine(alpaca_config, trading_config, model_config, sentiment_config, system_config)

async def train():
    await engine._weekend_prep()
    print("Crypto ML training complete — models saved to models/")

asyncio.run(train())
