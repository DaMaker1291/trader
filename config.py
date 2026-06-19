import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AlpacaConfig:
    api_key: str = os.getenv("ALPACA_API_KEY", "")
    secret_key: str = os.getenv("ALPACA_SECRET_KEY", "")
    base_url: str = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    data_url: str = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
    websocket_url: str = os.getenv("ALPACA_WS_URL", "wss://stream.data.alpaca.markets/v2/iex")


@dataclass
class TradingConfig:
    symbols: list[str] = None
    starting_cash: float = 100.0
    max_position_pct: float = 1.0
    max_total_exposure_pct: float = 100.0
    atr_stop_loss_multiplier: float = 0.0
    atr_take_profit_multiplier: float = 0.0
    min_atr_threshold: float = 0.0
    sentiment_weight: float = 0.0
    momentum_weight: float = 1.0
    prediction_threshold: float = 0.0
    min_bars_for_indicators: int = 1
    retrain_interval_bars: int = 5
    paper_trading: bool = True
    dry_run: bool = False
    crypto_mode: bool = False

    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["UPRO", "TQQQ", "SOXL", "FAS", "LABU", "BTC/USD", "ETH/USD", "SOL/USD"]


@dataclass
class ModelConfig:
    rf_n_estimators: int = 100
    rf_max_depth: int = 12
    rf_min_samples_split: int = 2
    sliding_window_size: int = 500
    feature_lookback: int = 20
    retrain_threshold: float = 0.005


@dataclass
class SentimentConfig:
    model_name: str = ""
    batch_size: int = 32
    max_seq_length: int = 128
    confidence_threshold: float = 0.3
    news_lookback_hours: int = 24
    cache_ttl_seconds: int = 300


@dataclass
class SystemConfig:
    log_level: str = "INFO"
    log_file: str = "trading_engine.log"
    reconnect_delay: float = 2.0
    max_reconnect_attempts: int = 9999
    heartbeat_interval: int = 10
    order_timeout: float = 5.0
    websocket_ping_interval: int = 10
    websocket_ping_timeout: int = 5


alpaca_config = AlpacaConfig()
trading_config = TradingConfig()
model_config = ModelConfig()
sentiment_config = SentimentConfig()
system_config = SystemConfig()
