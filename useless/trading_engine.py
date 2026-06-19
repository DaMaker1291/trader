#!/usr/bin/env python3
"""
Multi-Modal Alpaca Trading Engine
==================================
Production-grade algorithmic trading system combining technical analysis,
NLP-driven news sentiment (FinBERT), and online ML predictions.
Designed for live and after-hours trading via the Alpaca Trade API.

Architecture
------------
  WebSocket Stream (bars/quotes/trades)
       |
  asyncio dispatch (non-blocking)
       |
  +-------+  +----------+  +---------+
  | Tech  |  | Sentiment|  | Micro   |
  | Calc  |  | Engine   |  | Structure|
  +-------+  +----------+  +---------+
       |           |            |
       +-----------+------------+
                   |
           Feature Matrix
         (numerical + text-derived)
                   |
            ML Predictor
         (Random Forest / LSTM)
                   |
            Trading Signal
                   |
            Risk Manager
                   |
            Order Execution
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import signal
import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from functools import partial
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

import numpy as np
import pandas as pd

from config import (
    alpaca_config,
    trading_config,
    model_config,
    sentiment_config,
    system_config,
    AlpacaConfig,
    TradingConfig,
    ModelConfig,
    SentimentConfig,
    SystemConfig,
)

# ---------------------------------------------------------------------------
# Alpaca SDK
# ---------------------------------------------------------------------------
try:
    from alpaca.data.enums import DataFeed
    from alpaca.data.live import StockDataStream
    from alpaca.data.models import Bar, Quote, Trade, Orderbook
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, OrderStatus
    from alpaca.trading.requests import (
        GetOrdersRequest,
        MarketOrderRequest,
        LimitOrderRequest,
        StopLossRequest,
        TakeProfitRequest,
    )
    from alpaca.common.exceptions import APIError

    _ALPACA_AVAILABLE = True
except ImportError:
    _ALPACA_AVAILABLE = False

# ---------------------------------------------------------------------------
# Machine Learning
# ---------------------------------------------------------------------------
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

# ---------------------------------------------------------------------------
# Deep Learning (optional LSTM skeleton)
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Transformers / FinBERT
# ---------------------------------------------------------------------------
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

# ============================================================================
# Logging Configuration
# ============================================================================
log_level = getattr(logging, system_config.log_level.upper(), logging.INFO)
logger = logging.getLogger("MultiModalEngine")
logger.setLevel(log_level)

_fh = logging.FileHandler(system_config.log_file)
_fh.setLevel(log_level)
_ch = logging.StreamHandler(sys.stdout)
_ch.setLevel(log_level)
_fmt = logging.Formatter(
    "%(asctime)s.%(msecs)03d | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_fh.setFormatter(_fmt)
_ch.setFormatter(_fmt)
logger.addHandler(_fh)
logger.addHandler(_ch)

# ============================================================================
# Custom Exceptions
# ============================================================================


class TradingEngineError(Exception):
    """Base exception for the trading engine."""


class ConfigurationError(TradingEngineError):
    """Raised when required configuration is missing or invalid."""


class ConnectionError(TradingEngineError):
    """Raised when WebSocket or API connection fails."""


class RiskCheckFailed(TradingEngineError):
    """Raised when a trade is rejected by risk management."""


class OrderExecutionError(TradingEngineError):
    """Raised when an order fails to submit or fill."""


# ============================================================================
# Constants
# ============================================================================

US_EASTERN = timezone(timedelta(hours=-4))  # EDT
US_EASTERN_STD = timezone(timedelta(hours=-5))  # EST

PRE_MARKET_START = timedelta(hours=4, minutes=0)  # 04:00 ET
REGULAR_START = timedelta(hours=9, minutes=30)  # 09:30 ET
REGULAR_END = timedelta(hours=16, minutes=0)  # 16:00 ET
AFTER_HOURS_END = timedelta(hours=20, minutes=0)  # 20:00 ET


def _us_eastern_now() -> datetime:
    return datetime.now(US_EASTERN)


def _time_to_tod(time: datetime) -> timedelta:
    return timedelta(hours=time.hour, minutes=time.minute, seconds=time.second)


def is_extended_hours(now: Optional[datetime] = None) -> bool:
    """Return True if the market is in pre-market or after-hours session."""
    if now is None:
        now = _us_eastern_now()
    tod = _time_to_tod(now)
    weekday = now.weekday()
    if weekday >= 5:
        return False
    return tod < REGULAR_START or tod > REGULAR_END


def is_market_open(now: Optional[datetime] = None) -> bool:
    """Return True if regular or extended hours are active."""
    if now is None:
        now = _us_eastern_now()
    tod = _time_to_tod(now)
    weekday = now.weekday()
    if weekday >= 5:
        return False
    return PRE_MARKET_START <= tod <= AFTER_HOURS_END


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class MarketState:
    """Rolling state of a single symbol tracked by the engine."""

    symbol: str
    current_price: float = 0.0
    vwap: float = 0.0
    bid_price: float = 0.0
    ask_price: float = 0.0
    bid_size: float = 0.0
    ask_size: float = 0.0
    last_update: datetime = field(default_factory=lambda: datetime.min.replace(tzinfo=timezone.utc))

    # Rolling price buffers for indicator computation
    closes: Deque[float] = field(default_factory=lambda: deque(maxlen=200))
    highs: Deque[float] = field(default_factory=lambda: deque(maxlen=200))
    lows: Deque[float] = field(default_factory=lambda: deque(maxlen=200))
    volumes: Deque[float] = field(default_factory=lambda: deque(maxlen=200))
    timestamps: Deque[datetime] = field(default_factory=lambda: deque(maxlen=200))

    # Computed indicators
    ema_9: float = 0.0
    ema_21: float = 0.0
    rsi: float = 50.0
    macd_line: float = 0.0
    signal_line: float = 0.0
    macd_histogram: float = 0.0
    atr: float = 0.0

    # Micro-structure
    order_book_imbalance: float = 0.0

    # Sentiment (last computed)
    sentiment_score: float = 0.0
    sentiment_confidence: float = 0.0

    # ML prediction
    ml_prediction: float = 0.5
    ml_probability: float = 0.5

    def update_price_buffers(self, bar: Bar) -> None:
        """Append bar data to rolling deques."""
        self.closes.append(bar.close)
        self.highs.append(bar.high)
        self.lows.append(bar.low)
        self.volumes.append(bar.volume)
        self.timestamps.append(bar.timestamp)
        self.current_price = bar.close
        self.last_update = bar.timestamp

    def update_quote(self, quote: Quote) -> None:
        """Update best bid/ask from quote."""
        self.bid_price = quote.bid_price
        self.ask_price = quote.ask_price
        self.bid_size = float(quote.bid_size)
        self.ask_size = float(quote.ask_size)
        mid = (quote.bid_price + quote.ask_price) / 2.0
        if mid > 0:
            self.current_price = mid
        self.last_update = quote.timestamp
        self._compute_imbalance()

    def _compute_imbalance(self) -> None:
        """Compute order book imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol)."""
        denom = self.bid_size + self.ask_size
        if denom > 0:
            self.order_book_imbalance = (self.bid_size - self.ask_size) / denom
        else:
            self.order_book_imbalance = 0.0

    def is_ready(self, min_bars: int = 50) -> bool:
        """Return True if we have enough bars for meaningful indicators."""
        return len(self.closes) >= min_bars


@dataclass
class OpenPosition:
    """Track an open position for risk management."""

    symbol: str
    side: OrderSide
    entry_price: float
    quantity: float
    atr_at_entry: float
    stop_loss_price: float
    take_profit_price: float
    entry_time: datetime

    @property
    def is_long(self) -> bool:
        return self.side == OrderSide.BUY

    def current_pnl_pct(self, current_price: float) -> float:
        if self.entry_price == 0:
            return 0.0
        if self.is_long:
            return (current_price - self.entry_price) / self.entry_price
        return (self.entry_price - current_price) / self.entry_price


@dataclass
class TradeRecord:
    """Record of a completed trade for progressive learning."""
    symbol: str
    entry_price: float
    exit_price: float
    profit_pct: float
    entry_time: datetime
    exit_time: datetime
    reason: str  # 'stop_loss', 'take_profit', 'trailing_stop', 'manual'


class TradeLedger:
    """Tracks trade history and learns which symbols perform best.

    Progressive improvement:
      - Records every closed trade with profit/loss
      - Tracks win rate per symbol (last 10 trades)
      - Flags symbols with 3+ consecutive losses to avoid
      - Ranks symbols by average profit
    """

    def __init__(self, max_history: int = 500):
        self.trades: List[TradeRecord] = []
        self.max_history = max_history
        self._symbol_stats: Dict[str, Dict[str, Any]] = {}

    def record(self, trade: TradeRecord) -> None:
        self.trades.append(trade)
        if len(self.trades) > self.max_history:
            self.trades.pop(0)
        self._update_symbol_stats(trade.symbol)

    def _update_symbol_stats(self, symbol: str) -> None:
        sym_trades = [t for t in self.trades if t.symbol == symbol][-10:]
        if not sym_trades:
            return
        wins = sum(1 for t in sym_trades if t.profit_pct > 0)
        losses = sum(1 for t in sym_trades if t.profit_pct <= 0)
        avg_profit = sum(t.profit_pct for t in sym_trades) / len(sym_trades)
        self._symbol_stats[symbol] = {
            "trades": len(sym_trades),
            "wins": wins,
            "losses": losses,
            "win_rate": wins / max(len(sym_trades), 1),
            "avg_profit_pct": avg_profit,
            "consecutive_losses": self._consecutive_losses(sym_trades),
        }

    @staticmethod
    def _consecutive_losses(trades: List[TradeRecord]) -> int:
        count = 0
        for t in reversed(trades):
            if t.profit_pct <= 0:
                count += 1
            else:
                break
        return count

    def should_skip(self, symbol: str) -> bool:
        """Skip symbols with 3+ consecutive losses."""
        stats = self._symbol_stats.get(symbol)
        if stats is None:
            return False
        return stats["consecutive_losses"] >= 3

    def best_symbols(self, min_trades: int = 2) -> List[Tuple[str, float]]:
        """Return symbols sorted by avg profit, minimum trades filter."""
        scored = []
        for sym, s in self._symbol_stats.items():
            if s["trades"] >= min_trades:
                scored.append((sym, s["avg_profit_pct"]))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def summary(self) -> str:
        if not self.trades:
            return "No trades recorded"
        total = len(self.trades)
        wins = sum(1 for t in self.trades if t.profit_pct > 0)
        avg = sum(t.profit_pct for t in self.trades) / total
        return f"📊 TRADES: {total} | WINS: {wins} ({wins/total*100:.0f}%) | AVG: {avg:+.2f}%"


@dataclass
class TradingSignal:
    """Combined signal from all model components."""

    symbol: str
    direction: int  # 1 = long, -1 = short, 0 = none
    confidence: float
    ml_probability: float
    sentiment_score: float
    technical_score: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Technical Indicator Calculator
# ============================================================================


class TechnicalIndicatorCalculator:
    """On-the-fly computation of EMA, RSI, MACD, ATR, and VWAP features.

    All methods are stateless; feed them the rolling deques from MarketState.
    """

    @staticmethod
    def ema(values: Deque[float], period: int) -> float:
        if len(values) < period:
            return 0.0
        arr = list(values)
        k = 2.0 / (period + 1)
        result = sum(arr[:period]) / period
        for v in arr[period:]:
            result = v * k + result * (1 - k)
        return result

    @staticmethod
    def rsi(values: Deque[float], period: int = 14) -> float:
        if len(values) < period + 1:
            return 50.0
        arr = np.array(values, dtype=np.float64)
        deltas = np.diff(arr)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def macd(
        values: Deque[float],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> Tuple[float, float, float]:
        if len(values) < slow + signal:
            return 0.0, 0.0, 0.0
        ema_fast = TechnicalIndicatorCalculator.ema(values, fast)
        ema_slow = TechnicalIndicatorCalculator.ema(values, slow)
        macd_line = ema_fast - ema_slow
        # We need a separate deque of MACD values for the signal line
        # Approximate: use last `signal` computed MACD values from the available data
        macd_values = []
        for i in range(slow, len(values) + 1):
            sub_fast = TechnicalIndicatorCalculator.ema(
                deque(list(values)[:i]), fast
            )
            sub_slow = TechnicalIndicatorCalculator.ema(
                deque(list(values)[:i]), slow
            )
            macd_values.append(sub_fast - sub_slow)
        signal_line = sum(macd_values[-signal:]) / signal if len(macd_values) >= signal else 0.0
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def atr(highs: Deque[float], lows: Deque[float], closes: Deque[float], period: int = 14) -> float:
        if len(closes) < period + 1:
            return 0.0
        h = np.array(highs, dtype=np.float64)
        l_ = np.array(lows, dtype=np.float64)
        c = np.array(closes, dtype=np.float64)
        tr = np.maximum(
            h[1:] - l_[1:],
            np.maximum(
                np.abs(h[1:] - c[:-1]),
                np.abs(l_[1:] - c[:-1]),
            ),
        )
        return float(np.mean(tr[-period:]))

    @staticmethod
    def vwap(closes: Deque[float], volumes: Deque[float]) -> float:
        if not closes or not volumes:
            return 0.0
        c_arr = np.array(closes, dtype=np.float64)
        v_arr = np.array(volumes, dtype=np.float64)
        total_vol = v_arr.sum()
        if total_vol == 0:
            return 0.0
        return float(np.dot(c_arr, v_arr) / total_vol)

    @staticmethod
    def compute_all(state: MarketState) -> None:
        """Compute and update all indicators on the given MarketState in-place."""
        closes = state.closes
        highs = state.highs
        lows = state.lows
        volumes = state.volumes

        if len(closes) < 2:
            return

        state.ema_9 = TechnicalIndicatorCalculator.ema(closes, 9)
        state.ema_21 = TechnicalIndicatorCalculator.ema(closes, 21)
        state.rsi = TechnicalIndicatorCalculator.rsi(closes, 14)
        state.macd_line, state.signal_line, state.macd_histogram = (
            TechnicalIndicatorCalculator.macd(closes, 12, 26, 9)
        )
        state.atr = TechnicalIndicatorCalculator.atr(highs, lows, closes, 5)
        state.vwap = TechnicalIndicatorCalculator.vwap(closes, volumes)

    @staticmethod
    def vwap_deviation(state: MarketState) -> float:
        """Return (price - VWAP) / VWAP as a measure of deviation."""
        if state.vwap == 0 or state.current_price == 0:
            return 0.0
        return (state.current_price - state.vwap) / state.vwap

    @staticmethod
    def ema_crossover_strength(state: MarketState) -> float:
        """Normalized EMA(9) - EMA(21) difference as a fraction of price."""
        if state.current_price == 0:
            return 0.0
        return (state.ema_9 - state.ema_21) / state.current_price


# ============================================================================
# Sentiment Engine (FinBERT + News Aggregation)
# ============================================================================


class SentimentEngine:
    """Aggregate news headlines and compute real-time sentiment scores via FinBERT.

    Feature engineering note:
        Text-derived sentiment scores (-1 to +1) are merged with the numerical
        market vectors by treating them as additional feature columns. During
        inference, the sentiment score and confidence level are concatenated
        alongside technical indicators (RSI, MACD, ATR, etc.) and micro-structure
        features (order book imbalance, VWAP deviation) to form a unified
        multi-modal feature matrix. This allows the downstream ML model to learn
        cross-modal interactions, e.g., bullish technical patterns reinforced by
        positive news sentiment versus bearish divergence.
    """

    def __init__(self, config: SentimentConfig) -> None:
        self.config = config
        self._sentiment_pipeline: Optional[Any] = None
        self._cache: Dict[str, Tuple[float, float, float]] = {}  # headline -> (score, conf, ts)
        self._last_fetch: float = 0.0
        self._cached_aggregate: Tuple[float, float] = (0.0, 0.0)  # (avg_score, avg_confidence)
        self._news_buffer: List[Dict[str, Any]] = []

        self._init_model()

    def _init_model(self) -> None:
        """Load FinBERT model from HuggingFace."""
        model_name = self.config.model_name
        if not model_name:
            logger.info(
                "No HuggingFace model configured; SentimentEngine using heuristic fallback. "
                "Set SentimentConfig.model_name to enable (e.g., 'ProsusAI/finbert')."
            )
            return
        if not _HF_AVAILABLE:
            logger.warning(
                "transformers not available; SentimentEngine will use fallback "
                "dummy predictions. Install with: pip install transformers torch"
            )
            return
        try:
            logger.info("Loading model: %s", model_name)
            self._sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                max_length=self.config.max_seq_length,
                truncation=True,
                device=-1,  # CPU; set to 0 for GPU
            )
            logger.info("FinBERT model loaded successfully.")
        except Exception as exc:
            logger.error("Failed to load FinBERT: %s", exc)
            self._sentiment_pipeline = None

    def fetch_news(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Dummy news aggregation placeholder.

        In production, replace with Alpaca News API or NewsAPI.org calls.
        This method aggregates recent headlines for the given symbols and
        caches them for sentiment scoring.
        """
        _ = symbols  # placeholder; real impl would hit a REST endpoint

        # Simulate fetching news; in production, replace with:
        #   GET https://data.alpaca.markets/v1beta1/news?symbols=AAPL&limit=10
        dummy_news = [
            {
                "headline": f"{random.choice(['Bullish', 'Strong', 'Positive', 'Upbeat', 'Optimistic'])} "
                f"outlook for {s} driven by {random.choice(['earnings beat', 'product launch', 'analyst upgrade', 'strong demand'])}",
                "source": "DummyFeed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbols": [s],
            }
            for s in symbols
            for _ in range(random.randint(0, 2))
        ]
        self._news_buffer.extend(dummy_news)
        # Keep only recent news within lookback window
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.config.news_lookback_hours)
        self._news_buffer = [
            n
            for n in self._news_buffer
            if n.get("timestamp", "") >= cutoff.isoformat()
        ]
        return self._news_buffer

    def _score_text(self, headline: str) -> Tuple[float, float]:
        """Score a single headline with FinBERT; returns (score, confidence).

        Returns:
            score: float in [-1.0, +1.0] where +1 is positive, -1 negative.
            confidence: float in [0, 1] reflecting model certainty.
        """
        # Check cache
        cached = self._cache.get(headline)
        if cached and (time.time() - cached[2]) < self.config.cache_ttl_seconds:
            return cached[0], cached[1]

        if self._sentiment_pipeline is None:
            # Fallback: heuristic keyword scoring
            score = self._heuristic_score(headline)
            confidence = 0.5
            self._cache[headline] = (score, confidence, time.time())
            return score, confidence

        try:
            result = self._sentiment_pipeline(headline)[0]
            label = result["label"].lower()
            prob = result["score"]
            if "positive" in label:
                score = prob
            elif "negative" in label:
                score = -prob
            else:
                score = 0.0  # neutral
            confidence = prob
            self._cache[headline] = (score, confidence, time.time())
            return score, confidence
        except Exception as exc:
            logger.debug("FinBERT scoring failed for '%s': %s", headline[:40], exc)
            score = self._heuristic_score(headline)
            return score, 0.3

    @staticmethod
    def _heuristic_score(headline: str) -> float:
        """Simple keyword-based fallback when FinBERT is unavailable."""
        positive_words = {
            "bullish", "upgrade", "beat", "positive", "growth", "strong",
            "surge", "gain", "profit", "rally", "outperform", "buy", "optimistic",
            "momentum", "breakthrough", "innovation", "partnership", "launch",
        }
        negative_words = {
            "bearish", "downgrade", "miss", "negative", "decline", "weak",
            "drop", "loss", "fall", "sell", "underperform", "pessimistic",
            "downturn", "volatile", "risk", "lawsuit", "investigation", "recall",
        }
        words = set(headline.lower().split())
        pos_count = len(words & positive_words)
        neg_count = len(words & negative_words)
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        return (pos_count - neg_count) / total

    def compute_aggregate_sentiment(
        self, symbols: List[str]
    ) -> Tuple[float, float]:
        """Fetch news and compute aggregate sentiment score across symbols.

        Returns:
            (avg_sentiment, avg_confidence) where sentiment is in [-1, 1].
        """
        news_items = self.fetch_news(symbols)
        if not news_items:
            # Decay toward neutral when no news
            score, conf = self._cached_aggregate
            decayed = score * 0.95
            self._cached_aggregate = (decayed, conf * 0.95)
            return self._cached_aggregate

        total_score = 0.0
        total_conf = 0.0
        count = 0
        for item in news_items[-50:]:  # cap at 50 items
            headline = item.get("headline", "")
            if not headline:
                continue
            score, conf = self._score_text(headline)
            if conf >= self.config.confidence_threshold:
                total_score += score * conf
                total_conf += conf
                count += 1

        if count > 0 and total_conf > 0:
            avg_score = total_score / total_conf
            avg_conf = total_conf / count
        else:
            avg_score, avg_conf = 0.0, 0.0

        self._cached_aggregate = (avg_score, avg_conf)
        return avg_score, avg_conf


# ============================================================================
# Machine Learning Predictor
# ============================================================================


class MLPredictor:
    """Sliding-window predictor consuming multi-modal features.

    The feature matrix is constructed by horizontally concatenating:
      - Numerical market vectors: RSI, EMA diff, MACD hist, ATR%, VWAP dev,
        order book imbalance, volume ratio, spread %, price change %
      - Text-derived sentiment: FinBERT score, confidence, news volume

    This creates a unified (n_samples, n_features) array where the model
    can learn non-linear interactions between price action and news tone.

    The target is a binary label: 1 if next-bar close > current close, else 0.
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.feature_window: Deque[np.ndarray] = deque(maxlen=config.sliding_window_size)
        self.label_window: Deque[int] = deque(maxlen=config.sliding_window_size)
        self._model: Optional[RandomForestClassifier] = None
        self._scaler: StandardScaler = StandardScaler()
        self._is_fitted: bool = False
        self._bars_since_train: int = 0
        self._feature_names: List[str] = []

        self._init_model()

    def _init_model(self) -> None:
        if not _SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available; ML Predictor disabled.")
            return
        self._model = RandomForestClassifier(
            n_estimators=self.config.rf_n_estimators,
            max_depth=self.config.rf_max_depth,
            min_samples_split=self.config.rf_min_samples_split,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
            verbose=0,
        )
        logger.info(
            "MLPredictor initialized: RF(%d trees, max_depth=%d, window=%d)",
            self.config.rf_n_estimators,
            self.config.rf_max_depth,
            self.config.sliding_window_size,
        )

    @staticmethod
    def build_feature_vector(state: MarketState) -> np.ndarray:
        """Construct the multi-modal feature vector for a single MarketState.

        Feature layout (16 dimensions):
          [0]  RSI                         (0-100)
          [1]  EMA(9)-EMA(21) / price      (normalized crossover)
          [2]  MACD histogram              (scaled by price)
          [3]  ATR / price                 (volatility ratio)
          [4]  (price - VWAP) / VWAP       (VWAP deviation)
          [5]  Order book imbalance        (-1 to +1)
          [6]  Sentiment score             (-1 to +1)
          [7]  Sentiment confidence        (0 to 1)
          [8]  (ask - bid) / mid price     (spread %)
          [9]  Current volume / avg(vol)   (volume ratio)
          [10] (close - open) / open       (bar change %)
          [11] (high - low) / close        (range %)
          [12] Price / SMA(20) - 1         (distance from mean)
          [13] 1 if close > EMA(9) else 0  (trend filter)
          [14] 1 if EMA(9) > EMA(21) else 0 (trend confirmation)
          [15] log(volume + 1)             (volume magnitude)
        """
        p = state.current_price if state.current_price != 0 else 1e-8
        vwap = state.vwap if state.vwap != 0 else p
        avg_vol = np.mean(state.volumes) if state.volumes else 1.0
        sma20 = np.mean(state.closes) if state.closes else p

        spread = (state.ask_price - state.bid_price) / (state.ask_price + state.bid_price + 1e-8)
        bar_change = 0.0
        bar_range = 0.0
        if len(state.closes) >= 2:
            closes_arr = list(state.closes)
            bar_change = (closes_arr[-1] - closes_arr[-2]) / (closes_arr[-2] + 1e-8)
        if state.highs and state.lows:
            bar_range = (list(state.highs)[-1] - list(state.lows)[-1]) / p

        features = np.array(
            [
                state.rsi / 100.0,
                (state.ema_9 - state.ema_21) / p,
                state.macd_histogram / p,
                state.atr / p,
                (state.current_price - vwap) / vwap,
                state.order_book_imbalance,
                state.sentiment_score,
                state.sentiment_confidence,
                spread,
                state.volumes[-1] / (avg_vol + 1e-8) if state.volumes else 0.0,
                bar_change,
                bar_range,
                (p / sma20) - 1.0 if sma20 > 0 else 0.0,
                1.0 if state.ema_9 > 0 and p > state.ema_9 else 0.0,
                1.0 if state.ema_9 > state.ema_21 else 0.0,
                math.log1p(float(np.mean(state.volumes)) if state.volumes else 1.0),
            ],
            dtype=np.float64,
        )
        return features

    def add_sample(self, features: np.ndarray, label: int) -> None:
        """Add a feature vector and its label to the sliding window."""
        self.feature_window.append(features)
        self.label_window.append(label)
        self._bars_since_train += 1

    def _should_retrain(self) -> bool:
        """Check if model should be retrained based on new data volume."""
        if not self._is_fitted:
            return len(self.feature_window) >= 100
        return self._bars_since_train >= self.config.retrain_threshold * len(self.feature_window)

    def train(self) -> bool:
        """Train or retrain the Random Forest on the current sliding window.

        Returns True if training occurred.
        """
        if not self._model is None and not _SKLEARN_AVAILABLE:
            return False
        if len(self.feature_window) < max(50, self.config.rf_min_samples_split * 2):
            logger.debug("Not enough samples to train ML model: %d", len(self.feature_window))
            return False

        X = np.vstack(self.feature_window)
        y = np.array(self.label_window)

        # Scale features
        if not self._is_fitted:
            X_scaled = self._scaler.fit_transform(X)
        else:
            if hasattr(self._scaler, "partial_fit"):
                self._scaler.partial_fit(X)
                X_scaled = self._scaler.transform(X)
            else:
                X_scaled = self._scaler.fit_transform(X)

        self._model.fit(X_scaled, y)
        self._is_fitted = True
        self._bars_since_train = 0

        train_acc = self._model.score(X_scaled, y)
        logger.info(
            "ML model retrained on %d samples. Train accuracy: %.4f",
            len(self.feature_window),
            train_acc,
        )
        return True

    def predict(self, features: np.ndarray) -> Tuple[float, float]:
        """Predict probability of upward movement.

        Returns:
            (prediction_class, probability_up)
              prediction_class: 0 or 1
              probability_up: float in [0, 1]
        """
        if not self._is_fitted or self._model is None:
            return 0, 0.5

        try:
            X = features.reshape(1, -1)
            X_scaled = self._scaler.transform(X)
            proba = self._model.predict_proba(X_scaled)[0]
            # proba[0] = prob(down), proba[1] = prob(up)
            if len(proba) == 2:
                pred = 1 if proba[1] > 0.5 else 0
                return float(pred), float(proba[1])
            return 0, 0.5
        except Exception as exc:
            logger.warning("ML prediction failed: %s", exc)
            return 0, 0.5

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Return feature importance mapping if model is trained."""
        if not self._is_fitted or not hasattr(self._model, "feature_importances_"):
            return None
        names = self._feature_names or [f"f{i}" for i in range(16)]
        importances = self._model.feature_importances_
        return dict(sorted(zip(names, importances), key=lambda x: -x[1]))


# ============================================================================
# LSTM Skeleton (Optional PyTorch Module)
# ============================================================================


class LSTMPredictorSkeleton(nn.Module):
    """Lightweight LSTM for next-bar direction prediction.

    This is a skeleton that can be swapped in for the Random Forest when
    sufficient historical data is available. It consumes the same multi-modal
    feature vector (16 dims) and outputs a binary probability.

    Architecture:
        Linear(16 -> 64) -> LSTM(64, 32, 2 layers) -> Dropout(0.2) -> Linear(32 -> 1) -> Sigmoid
    """

    def __init__(self, input_dim: int = 16, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(hidden_dim // 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_dim)
        x = torch.relu(self.input_proj(x))
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        out = self.dropout(last_out)
        out = self.output(out)
        return self.sigmoid(out).squeeze(-1)


class LSTMTrainer:
    """Minimal training loop for the LSTM predictor."""

    def __init__(self, model: nn.Module, lr: float = 1e-3):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.BCELoss()

    def train_step(self, X: torch.Tensor, y: torch.Tensor) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        preds = self.model(X)
        loss = self.criterion(preds, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()


# ============================================================================
# Risk Manager
# ============================================================================


class RiskManager:
    """Strict dynamic risk management using ATR-based sizing and capital preservation.

    Core rules:
      - Position sizing: 2-5% of portfolio per trade, scaled by ATR volatility.
      - Stop loss: entry_price - stop_loss_mult * ATR (long) or + mult (short).
      - Take profit: entry_price + take_profit_mult * ATR (long) or - mult (short).
      - Max concurrent exposure: 50% of portfolio.
      - Minimum ATR threshold: avoid trading in insufficiently volatile conditions.
    """

    def __init__(self, config: TradingConfig) -> None:
        self.config = config
        self.positions: Dict[str, OpenPosition] = {}

    def compute_position_size(
        self,
        symbol: str,
        atr: float,
        current_price: float,
        portfolio_value: float,
    ) -> Tuple[float, float]:
        """Compute position allocation as (notional_amount, quantity).

        Uses notional (dollar) sizing for maximum capital efficiency.
        For small accounts, this enables fractional share ordering.

        Returns:
            (notional_amount, quantity) where notional is the dollar amount
            to allocate, and quantity is notional / current_price (may be fractional).
        """
        if current_price <= 0 or portfolio_value <= 0:
            return 0.0, 0.0

        max_allocation = portfolio_value * self.config.max_position_pct

        # Volatility-adjusted notional
        if atr > 0:
            vol_adj = max_allocation / (atr * current_price + 1e-10)
        else:
            vol_adj = max_allocation

        notional = min(max_allocation, vol_adj)
        quantity = notional / current_price

        # Ensure minimum meaningful quantity
        if quantity < 0.0001:
            return 0.0, 0.0

        return float(notional), float(quantity)

    def compute_stop_loss(self, entry_price: float, atr: float, side: OrderSide) -> float:
        """ATR-based stop loss."""
        if side == OrderSide.BUY:
            return entry_price - self.config.atr_stop_loss_multiplier * atr
        return entry_price + self.config.atr_stop_loss_multiplier * atr

    def compute_take_profit(self, entry_price: float, atr: float, side: OrderSide) -> float:
        """ATR-based take profit."""
        if side == OrderSide.BUY:
            return entry_price + self.config.atr_take_profit_multiplier * atr
        return entry_price - self.config.atr_take_profit_multiplier * atr

    def can_open_new_position(
        self,
        symbol: str,
        portfolio_value: float,
        position_value: float,
        atr: float,
        current_price: float,
    ) -> Tuple[bool, str]:
        """Check risk constraints before opening a new position.

        Returns (allowed, reason_string).
        """
        if symbol in self.positions:
            return False, f"Position already open for {symbol}"

        if position_value > portfolio_value * self.config.max_total_exposure_pct:
            return False, "Max total exposure exceeded"

        if atr > 0 and atr < self.config.min_atr_threshold * current_price:
            return False, f"ATR too low for meaningful volatility ({atr:.4f})"

        if portfolio_value <= 0:
            return False, "Portfolio value is zero or negative"

        return True, "OK"

    def check_stop_loss_take_profit(
        self,
        symbol: str,
        current_price: float,
    ) -> Tuple[bool, bool]:
        """Check if position should be stopped out or take profit hit.

        Returns (stop_loss_hit, take_profit_hit).
        """
        pos = self.positions.get(symbol)
        if pos is None:
            return False, False

        if pos.is_long:
            stop_hit = current_price <= pos.stop_loss_price
            tp_hit = current_price >= pos.take_profit_price
        else:
            stop_hit = current_price >= pos.stop_loss_price
            tp_hit = current_price <= pos.take_profit_price

        return stop_hit, tp_hit


# ============================================================================
# Trade-to-Bar Aggregator (for after-hours / extended hours trading)
# ============================================================================


class TradeBarAggregator:
    """Aggregate individual trades into synthetic 1-minute bars.

    During after-hours/extended-hours, Alpaca does not publish bar data.
    This class accumulates trade prints and emits a synthetic Bar when
    the aggregation window (wall-clock time) elapses.
    """

    def __init__(self, symbol: str, window_seconds: int = 60) -> None:
        self.symbol = symbol
        self.window_seconds = window_seconds
        self.reset()

    def reset(self) -> None:
        self.open: float = 0.0
        self.high: float = 0.0
        self.low: float = float("inf")
        self.close: float = 0.0
        self.volume: float = 0.0
        self.trade_count: int = 0
        self.vwap_numerator: float = 0.0
        self.window_start: Optional[datetime] = None
        self._last_trade_time: Optional[datetime] = None

    def add_trade(self, trade: Trade) -> Optional[Bar]:
        """Add a trade and return a synthetic Bar if the window has elapsed."""
        price = float(trade.price)
        size = float(trade.size)
        ts = trade.timestamp

        if self.window_start is None:
            self.window_start = ts
            self.open = price

        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume += size
        self.trade_count += 1
        self.vwap_numerator += price * size
        self._last_trade_time = ts

        elapsed = (ts - self.window_start).total_seconds()
        if elapsed >= self.window_seconds:
            return self._emit_bar(ts)
        return None

    def _emit_bar(self, ts: datetime) -> Bar:
        vwap = self.vwap_numerator / self.volume if self.volume > 0 else self.close
        raw = {
            "s": self.symbol,
            "t": ts.isoformat(),
            "o": self.open,
            "h": self.high,
            "l": self.low,
            "c": self.close,
            "v": self.volume,
            "n": self.trade_count,
            "vw": vwap,
        }
        bar = Bar(symbol=self.symbol, raw_data=raw)
        self.reset()
        return bar


# ============================================================================
# Momentum Scanner — finds the day's strongest movers
# ============================================================================

# yfinance for real-time market scanning (free, no API key needed)
try:
    import yfinance as yf
    _YFINANCE_AVAILABLE = True
except ImportError:
    _YFINANCE_AVAILABLE = False


class MomentumScanner:
    """Scans a universe of liquid stocks/ETFs and returns the top gainers by
    daily percent change.

    Data sources (tried in order):
      1. yfinance (Yahoo Finance) — free, real-time, no API key
      2. Alpaca snapshot REST API — fallback when yfinance is unavailable

    The goal: catch 'free risers' — stocks with strong upward momentum
    that are likely to continue gaining intraday.
    """

    # Symbols subscribed on IEX. Free tier limit is ~5-10 symbols.
    # Leveraged ETFs are the core — their 3x leverage is our primary strategy.
    # For scanner picks (AMC, DKNG, etc.), we get price via yfinance.
    BROAD_SUBSCRIBE: List[str] = [
        "UPRO", "TQQQ", "SOXL", "FAS", "LABU",
        "SPY", "QQQ",  # indices for context
    ]

    # ~150 liquid symbols spanning sectors, leveraged ETFs, and high-beta names
    # Crypto scan universe (full list for yfinance scanning)
    CRYPTO_SYMBOLS: List[str] = [
        "BTC-USD", "ETH-USD", "SOL-USD",
        "XRP-USD", "DOGE-USD", "ADA-USD", "AVAX-USD", "DOT-USD",
        "LINK-USD", "UNI-USD", "ATOM-USD", "LTC-USD",
        "BCH-USD", "NEAR-USD", "APT-USD", "SUI-USD", "PEPE-USD",
        "WIF-USD", "INJ-USD", "TIA-USD", "SEI-USD",
    ]
    # Pairs actually tradeable on Alpaca paper (for order submission)
    ALPACA_CRYPTO: List[str] = ["BTC-USD", "ETH-USD", "SOL-USD"]

    UNIVERSE: List[str] = [
        # --- Major Indices & Sector ETFs ---
        "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "VOOG", "VBK",
        "XLF", "XLK", "XLV", "XLE", "XLI", "XLP", "XLU", "XLY", "XLB", "XLRE",
        "ARKK", "ARKQ", "ARKW", "ARKG", "ARKF",
        # --- Leveraged ETFs ---
        "UPRO", "TQQQ", "SOXL", "FAS", "LABU", "SPXL", "TECL", "FNGU",
        "UDOW", "URTY", "YINN", "CURE", "RETL", "DPST", "JNUG", "NUGT",
        # --- Mega-cap Tech ---
        "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA",
        # --- AI / Software / Cybersecurity ---
        "PLTR", "SOUN", "IONQ", "RGTI", "ARM", "CRWD", "PANW", "ZS",
        "DDOG", "MDB", "SNOW", "NET", "AI", "BBAI", "SOUN",
        # --- Crypto-exposed ---
        "COIN", "MSTR", "RIOT", "CLSK", "IBIT", "BITO", "MARA",
        # --- Semiconductors ---
        "AMD", "INTC", "MU", "AVGO", "QCOM", "MRVL", "AMAT", "LRCX",
        "KLAC", "NXPI", "STM", "TSM", "ASML", "ON",
        # --- Biotech / Pharma ---
        "LLY", "UNH", "JNJ", "PFE", "ABBV", "BIIB", "VRTX", "MRNA",
        "REGN", "GILD", "AMGN", "BMY", "TEVA",
        # --- Financials ---
        "JPM", "GS", "BAC", "MS", "V", "MA", "AXP", "SCHW", "BLK",
        "C", "WFC", "USB", "PNC",
        # --- Consumer / Retail ---
        "WMT", "COST", "HD", "LOW", "NFLX", "DIS", "SBUX", "MCD",
        "CMG", "TSCO", "ROST", "TJX", "TGT",
        # --- Growth / SaaS ---
        "CRM", "NOW", "ADBE", "ORCL", "INTU", "UBER", "DASH", "LYFT",
        "SNAP", "PINS", "RBLX", "ZM", "WDAY", "TEAM",
        # --- Industrials / Defense ---
        "LMT", "RTX", "NOC", "GD", "LHX", "GE", "BA", "CAT", "DE",
        # --- Clean Energy / Commodities ---
        "TAN", "ICLN", "ENPH", "SEDG", "FSLR", "XLE", "USO", "GDX",
        "GDXJ", "SLV", "GLD",
        # --- High-volatility / Meme ---
        "HOOD", "CHWY", "DKNG", "RIVN", "LCID", "AMC", "GME", "CVNA",
        "CELH", "CROX", "AFRM", "SOFI", "UPST",
        # --- Healthcare ---
        "ISRG", "SYK", "MDT", "BSX", "ABT", "DHR", "TMO", "IQV",
        # --- Communication Services ---
        "GOOGL", "META", "NFLX", "DIS", "CMCSA", "CHTR", "T", "VZ",
        # --- Transports ---
        "IYT", "FDX", "UPS", "JBHT", "SAIA", "ODFL",
        # --- Regional Banks (high beta) ---
        "KRE", "KBE", "HBAN", "KEY", "ZION", "FITB", "RF",
        # --- REITs ---
        "PLD", "AMT", "CCI", "EQIX", "DLR", "O", "WELL",
        # --- International ---
        "EEM", "VWO", "FXI", "EWJ", "EWZ", "INDA", "RSX",
        # --- Bonds (for hedging scans) ---
        "TLT", "IEF", "SHY", "AGG", "BND", "LQD", "HYG",
        # --- Crypto (24/7 trading pairs) ---
        "BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "ADA-USD",
        "XRP-USD", "DOT-USD", "LINK-USD", "AVAX-USD",
    ]

    def __init__(self, api_key: str, secret_key: str) -> None:
        self.api_key = api_key
        self.secret_key = secret_key
        self._historical_client: Optional[Any] = None

    # ------------------------------------------------------------------
    # yfinance Scanner (primary — free, real-time, no API key)
    # ------------------------------------------------------------------

    async def scan_via_yfinance(
        self, n: int = 5, min_volume: int = 50000, min_price: float = 1.0,
        symbols: Optional[List[str]] = None,
    ) -> List[Tuple[str, float, float, float, float]]:
        """Scan using Yahoo Finance. Returns top N gainers.

        Uses batch download for speed: fetches 2 daily bars per symbol
        and calculates today's change %.

        Args:
            n: Number of top gainers to return.
            min_volume: Minimum volume filter.
            min_price: Minimum price filter.
            symbols: Subset to scan (None = full universe).

        Returns:
            List of (symbol, change_pct, volume, close_price, prev_close) sorted by change descending.
        """
        if not _YFINANCE_AVAILABLE:
            logger.debug("yfinance not available; falling back to Alpaca scanner")
            return []

        try:
            scan_list = symbols if symbols is not None else self.UNIVERSE
            batch_size = 50
            all_gainers: List[Tuple[str, float, float, float, float]] = []

            for i in range(0, len(scan_list), batch_size):
                batch = scan_list[i:i + batch_size]
                try:
                    # Run in executor to avoid blocking the event loop
                    loop = asyncio.get_running_loop()
                    data = await loop.run_in_executor(
                        None,
                        lambda tickers=batch: yf.download(
                            tickers=" ".join(tickers),
                            period="2d",
                            interval="1d",
                            progress=False,
                            auto_adjust=True,
                            prepost=False,
                            group_by="ticker",
                        ),
                    )

                    if data is None or data.empty:
                        continue

                    if isinstance(data.columns, pd.MultiIndex):
                        tickers_found = data.columns.get_level_values(0).unique()
                    else:
                        tickers_found = batch

                    for sym in batch:
                        try:
                            if isinstance(data.columns, pd.MultiIndex):
                                if sym not in tickers_found:
                                    continue
                                close = data[(sym, "Close")]
                                vol_series = data[(sym, "Volume")]
                            else:
                                close = data["Close"]
                                vol_series = data["Volume"]

                            if len(close) < 2:
                                continue
                            today = float(close.iloc[-1])
                            yesterday = float(close.iloc[-2])
                            vol = int(vol_series.iloc[-1])

                            change_pct = ((today - yesterday) / yesterday) * 100
                            if change_pct <= 0 or vol < min_volume or today < min_price:
                                continue
                            all_gainers.append((sym, change_pct, vol, today, yesterday))

                        except (KeyError, IndexError, TypeError, ValueError) as exc:
                            logger.debug("yfinance: skipped %s: %s", sym, exc)
                            continue

                except Exception as exc:
                    logger.debug("yfinance batch scan failed: %s", exc)
                    continue

            all_gainers.sort(key=lambda x: x[1], reverse=True)
            return all_gainers[:n]

        except Exception as exc:
            logger.warning("yfinance scanner error: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Alpaca Snapshot Scanner (fallback)
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._historical_client is None:
            from alpaca.data.historical import StockHistoricalDataClient
            self._historical_client = StockHistoricalDataClient(
                self.api_key, self.secret_key
            )
        return self._historical_client

    async def scan_top_gainers(
        self, n: int = 5, min_volume: int = 50000, min_price: float = 2.0
    ) -> List[Tuple[str, float, float]]:
        """Fetch snapshots for the universe and return the top N gainers.

        Returns:
            List of (symbol, daily_change_pct, volume) sorted by change descending.
        """
        client = self._get_client()
        all_snapshots: Dict[str, Any] = {}

        # Alpaca batch limit is 100 symbols per request
        batch_size = 100
        for i in range(0, len(self.UNIVERSE), batch_size):
            batch = self.UNIVERSE[i:i + batch_size]
            try:
                snapshots = client.get_stock_snapshots(symbols=batch)
                if snapshots:
                    all_snapshots.update(snapshots)
            except Exception as exc:
                logger.debug("Snapshot batch failed for %d symbols: %s", len(batch), exc)

        gainers: List[Tuple[str, float, float]] = []
        for sym, snap in all_snapshots.items():
            if snap is None:
                continue
            try:
                daily = snap.daily_bar
                if daily is None:
                    continue
                change_pct = ((daily.close - daily.open) / daily.open) * 100
                volume = daily.volume or 0
                if change_pct < 0 or volume < min_volume:
                    continue
                gainers.append((sym, change_pct, volume))
            except (AttributeError, TypeError, ZeroDivisionError):
                continue

        gainers.sort(key=lambda x: x[1], reverse=True)
        return gainers[:n]

    async def scan_top_momentum(
        self, n: int = 5, min_volume: int = 50000
    ) -> List[Tuple[str, float, float, float]]:
        """Enhanced scan: combines daily change with volume and price.

        Returns:
            List of (symbol, momentum_score, change_pct, volume).
            momentum_score = change_pct * log(volume + 1) / 100
        """
        client = self._get_client()
        all_snapshots: Dict[str, Any] = {}

        batch_size = 100
        for i in range(0, len(self.UNIVERSE), batch_size):
            batch = self.UNIVERSE[i:i + batch_size]
            try:
                snapshots = client.get_stock_snapshots(symbols=batch)
                if snapshots:
                    all_snapshots.update(snapshots)
            except Exception as exc:
                logger.debug("Snapshot batch failed: %s", exc)

        scored: List[Tuple[str, float, float, float]] = []
        for sym, snap in all_snapshots.items():
            if snap is None:
                continue
            try:
                daily = snap.daily_bar
                latest = snap.latest_trade
                if daily is None or latest is None:
                    continue
                change_pct = ((daily.close - daily.open) / daily.open) * 100
                volume = daily.volume or 0
                price = latest.price or 0

                if change_pct < 0 or volume < min_volume or price < 1.0:
                    continue

                # Momentum score: % change weighted by volume
                momentum = change_pct * math.log1p(volume) / 100.0
                scored.append((sym, momentum, change_pct, volume))
            except (AttributeError, TypeError, ZeroDivisionError):
                continue

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]

    def get_trending_sectors(self, gainers: List[Tuple[str, float, float]]) -> Dict[str, int]:
        """Simple sector tagger — count how many gainers per implied sector."""
        sector_map: Dict[str, List[str]] = {
            "Tech": ["AAPL", "MSFT", "GOOGL", "META", "CRM", "NOW", "ADBE", "ORCL"],
            "AI": ["NVDA", "AMD", "PLTR", "SOUN", "ARM", "CRWD", "PANW", "AI"],
            "Semis": ["AMD", "INTC", "MU", "AVGO", "QCOM", "MRVL", "AMAT", "TSM", "ASML"],
            "Leveraged": ["UPRO", "TQQQ", "SOXL", "FAS", "LABU", "SPXL", "TECL", "FNGU"],
            "Crypto": ["COIN", "MSTR", "RIOT", "CLSK", "IBIT", "MARA"],
            "Biotech": ["LLY", "UNH", "ABBV", "VRTX", "MRNA", "REGN", "BIIB"],
            "Consumer": ["AMZN", "WMT", "COST", "TSLA", "NFLX", "DIS", "SBUX", "MCD"],
            "Financial": ["JPM", "GS", "BAC", "V", "MA", "AXP", "SCHW"],
        }
        syms = {s for s, _, _ in gainers}
        counts: Dict[str, int] = {}
        for sector, members in sector_map.items():
            c = len(syms & set(members))
            if c > 0:
                counts[sector] = c
        return counts


# ============================================================================
# Multi-Modal Trading Engine
# ============================================================================


class MultiModalTradingEngine:
    """Core trading engine orchestrating data ingestion, AI inference, and order execution.

    High-level flow:
      1. Connect to Alpaca WebSocket for bars/quotes/trades.
      2. For each incoming bar:
         a. Update technical indicators (EMA, RSI, MACD, ATR, VWAP).
         b. Compute micro-structure features (order book imbalance).
         c. Score recent news sentiment via FinBERT.
         d. Build multi-modal feature vector.
         e. Predict next-bar direction via ML model.
         f. If signal exceeds threshold + risk checks pass, execute trade.
      3. Monitor open positions for stop-loss / take-profit.
      4. Retrain ML model periodically on accumulated data.

    After-hours trading is supported via the SIP data feed and extended_hours=True
    flag on orders.
    """

    def __init__(
        self,
        alpaca_cfg: AlpacaConfig,
        trading_cfg: TradingConfig,
        model_cfg: ModelConfig,
        sentiment_cfg: SentimentConfig,
        sys_cfg: SystemConfig,
    ) -> None:
        self.alpaca_cfg = alpaca_cfg
        self.trading_cfg = trading_cfg
        self.model_cfg = model_cfg
        self.sentiment_cfg = sentiment_cfg
        self.sys_cfg = sys_cfg

        # Runtime state (must set before validation)
        self.symbols: List[str] = list(trading_cfg.symbols)
        self.is_running: bool = False
        self.is_dry_run: bool = trading_cfg.dry_run
        self.is_paper: bool = trading_cfg.paper_trading
        self.start_time: Optional[datetime] = None

        # --- Exit parameters (tuned per asset class) ---
        if self.trading_cfg.crypto_mode:
            from crypto_profile import (
                HARD_SL_PCT as CRYPTO_SL,
                TRAIL_ACT_PCT as CRYPTO_TRAIL_ACT,
                TRAIL_DIST_PCT as CRYPTO_TRAIL_DIST,
                SYMBOLS as CRYPTO_SYMBOLS_YF,
                HOT_SYMBOLS as CRYPTO_HOT_YF,
                ALPACA_SYMBOLS as CRYPTO_ALPACA_SYMBOLS,
                ML_DATA_PERIOD as CRYPTO_ML_PERIOD,
            )
            self.hard_sl_pct = CRYPTO_SL
            self.trail_act_pct = CRYPTO_TRAIL_ACT
            self.trail_dist_pct = CRYPTO_TRAIL_DIST
            self.symbols = list(CRYPTO_ALPACA_SYMBOLS)
            self._crypto_hot_symbols = list(CRYPTO_HOT_YF)       # yfinance format for scanning
            self._crypto_all_symbols = list(CRYPTO_SYMBOLS_YF)   # yfinance format for scanning
            self.ml_data_period = CRYPTO_ML_PERIOD
            self._crypto_mode = True
            logger.info("CRYPTO MODE: exit params (SL=%.1f%%, trail_act=%.1f%%, trail_dist=%.1f%%)",
                        self.hard_sl_pct, self.trail_act_pct, self.trail_dist_pct)
        else:
            self.hard_sl_pct = 0.7
            self.trail_act_pct = 0.5
            self.trail_dist_pct = 0.3
            self._crypto_hot_symbols = []
            self._crypto_all_symbols = []
            self.ml_data_period = "3mo"
            self._crypto_mode = False

        # Validate configuration
        self._validate_config()
        self.bar_count: int = 0
        self.total_trades: int = 0

        # Per-symbol market state
        self.market_states: Dict[str, MarketState] = {
            sym: MarketState(symbol=sym) for sym in self.symbols
        }

        # Component sub-systems
        self.technical_calc = TechnicalIndicatorCalculator()
        self.sentiment_engine = SentimentEngine(sentiment_cfg)
        self.ml_predictor = MLPredictor(model_cfg)
        self.risk_manager = RiskManager(trading_cfg)

        # After-hours trade aggregators (one per symbol)
        self._trade_aggregators: Dict[str, TradeBarAggregator] = {
            sym: TradeBarAggregator(sym) for sym in self.symbols
        }

        # LSTM - enabled by default for maximum predictive power
        self._lstm_model: Optional[nn.Module] = None
        self._lstm_trainer: Optional[LSTMTrainer] = None
        self._enable_lstm: bool = True if _TORCH_AVAILABLE else False
        if self._enable_lstm:
            logger.info("LSTM neural predictor ENABLED for enhanced signal processing")
            self._lstm_model = LSTMPredictorSkeleton(input_dim=16)
            self._lstm_trainer = LSTMTrainer(self._lstm_model)
            logger.info("LSTM model initialized: 16-input → 64-proj → LSTM(32) → 1-output")

        # Compounding performance tracker
        self._peak_equity: float = trading_cfg.starting_cash
        self._last_logged_equity: float = trading_cfg.starting_cash
        self._daily_trade_count: int = 0
        self._compounding_periods: int = 0
        self._total_return_pct: float = 0.0
        self._target_equity: float = 1_000_000_000.0  # $1 Billion

        # Momentum scanner for finding 'free risers'
        self._momentum_scanner = MomentumScanner(
            alpaca_cfg.api_key, alpaca_cfg.secret_key
        )
        self._scanned_symbols: List[str] = []  # symbols from today's scan
        self._last_scan_time: float = 0.0
        self._scan_interval: float = 300.0  # rescan every 5 min

        # Trade ledger for progressive learning
        self._ledger = TradeLedger()

        # Alpaca clients (lazy init)
        self._trading_client: Optional[TradingClient] = None
        self._stock_stream: Optional[StockDataStream] = None
        self._crypto_stream: Optional[Any] = None

        # WebSocket reconnect state
        self._reconnect_attempts: int = 0
        self._max_reconnect_attempts: int = sys_cfg.max_reconnect_attempts
        self._reconnect_delay: float = sys_cfg.reconnect_delay

        # Shutdown coordination
        self._shutdown_event: asyncio.Event = asyncio.Event()
        self._tasks: List[asyncio.Task] = []

        logger.info(
            "MultiModalTradingEngine initialized. Dry-run=%s, Paper=%s, Symbols=%s",
            self.is_dry_run,
            self.is_paper,
            self.symbols,
        )

    def _validate_config(self) -> None:
        """Validate critical configuration parameters."""
        if not self.alpaca_cfg.api_key or not self.alpaca_cfg.secret_key:
            raise ConfigurationError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in environment or config."
            )
        if not self.symbols:
            raise ConfigurationError("At least one trading symbol must be specified.")
        if self.trading_cfg.starting_cash <= 0:
            raise ConfigurationError("Starting cash must be positive.")

    # ------------------------------------------------------------------
    # Connection Management
    # ------------------------------------------------------------------

    def _init_clients(self) -> None:
        """Initialize Alpaca REST and WebSocket clients (stock + crypto)."""
        if not _ALPACA_AVAILABLE:
            raise ConfigurationError(
                "alpaca-py is not installed. Run: pip install alpaca-py"
            )

        self._stock_stream = StockDataStream(
            api_key=self.alpaca_cfg.api_key,
            secret_key=self.alpaca_cfg.secret_key,
            feed=DataFeed.IEX,
        )

        try:
            from alpaca.data.live.crypto import CryptoDataStream
            self._crypto_stream = CryptoDataStream(
                api_key=self.alpaca_cfg.api_key,
                secret_key=self.alpaca_cfg.secret_key,
            )
        except Exception:
            self._crypto_stream = None

        self._trading_client = TradingClient(
            api_key=self.alpaca_cfg.api_key,
            secret_key=self.alpaca_cfg.secret_key,
            paper=self.is_paper,
        )

        logger.info(
            "Clients initialized. IEX=%s, Crypto=%s, Paper=%s",
            self._stock_stream is not None,
            self._crypto_stream is not None,
            self.is_paper,
        )

    def _subscribe_handlers(self) -> None:
        """Wire up data stream subscriptions for stock + crypto symbols."""
        if self._stock_stream is None:
            self._init_clients()

        stock_syms = [s for s in self.symbols if "/" not in s]
        crypto_syms = [s for s in self.symbols if "/" in s]

        # Subscribe stocks (base + broad)
        all_stock = list(set(stock_syms + MomentumScanner.BROAD_SUBSCRIBE))
        if all_stock and self._stock_stream:
            self._stock_stream.subscribe_bars(self._on_bars, *all_stock)
            self._stock_stream.subscribe_quotes(self._on_quotes, *all_stock)
            self._stock_stream.subscribe_trades(self._on_trades, *all_stock)
        for sym in all_stock:
            if sym not in self.market_states:
                self.market_states[sym] = MarketState(symbol=sym)
                self._trade_aggregators[sym] = TradeBarAggregator(sym)

        # Subscribe crypto (24/7)
        if crypto_syms and self._crypto_stream is not None:
            self._crypto_stream.subscribe_bars(self._on_bars, *crypto_syms)
            self._crypto_stream.subscribe_quotes(self._on_quotes, *crypto_syms)
            self._crypto_stream.subscribe_trades(self._on_trades, *crypto_syms)
        for sym in crypto_syms:
            if sym not in self.market_states:
                self.market_states[sym] = MarketState(symbol=sym)
                self._trade_aggregators[sym] = TradeBarAggregator(sym)

        logger.info("Subscribed: %d stocks + %d crypto", len(all_stock), len(crypto_syms))

    # ------------------------------------------------------------------
    # Dynamic Symbol Management (Momentum Scanner Integration)
    # ------------------------------------------------------------------

    def _add_dynamic_symbols(self, new_symbols: List[str]) -> int:
        """Activate symbols for trading. Stock symbols need no sub (already subscribed).
        Crypto symbols get subscribed here.
        """
        added = 0
        for sym in new_symbols:
            if sym in self.market_states:
                continue
            self.market_states[sym] = MarketState(symbol=sym)
            self._trade_aggregators[sym] = TradeBarAggregator(sym)
            # Crypto streams need explicit subscription
            if "/" in sym and self._crypto_stream is not None:
                self._crypto_stream.subscribe_bars(self._on_bars, sym)
                self._crypto_stream.subscribe_quotes(self._on_quotes, sym)
                self._crypto_stream.subscribe_trades(self._on_trades, sym)
            added += 1

        if added > 0:
            logger.info("📡 ACTIVATED: %d new targets: %s", added, new_symbols[:10])
        return added

    async def _run_momentum_scan(self) -> None:
        """Scan the market for top gainers and dynamically subscribe to them.

        Runs at startup and periodically during market hours.
        Uses the MomentumScanner to find the strongest movers.
        """
        try:
            now_et = _us_eastern_now()
            weekday = now_et.weekday()

            # Only scan during market hours (pre-market 4:00 ET to 20:00 ET)
            tod = _time_to_tod(now_et)
            if weekday >= 5 or tod < PRE_MARKET_START or tod > AFTER_HOURS_END:
                logger.debug("Momentum scan skipped — market closed")
                return

            logger.info("🔍 MOMENTUM SCAN: scanning %d symbols for top gainers (yfinance)...",
                        len(MomentumScanner.UNIVERSE))

            # Try yfinance first (free, real-time), fall back to Alpaca snapshots
            top_raw = await self._momentum_scanner.scan_via_yfinance(n=5)
            if not top_raw:
                logger.info("yfinance returned no gainers; trying Alpaca snapshots...")
                top_raw = await self._momentum_scanner.scan_top_momentum(n=5)

            # Normalize to (symbol, change_pct, volume) format
            if top_raw and len(top_raw[0]) >= 4:
                top = [(s, chg, vol) for s, chg, vol, _, _ in top_raw]
            else:
                top = list(top_raw)  # already (symbol, change_pct, volume)

            # Convert yahoo-format crypto (BTC-USD) to Alpaca format (BTC/USD)
            converted = []
            for sym, chg, vol in top:
                if "-" in sym and sym.count("-") == 1 and sym.split("-")[1] in ("USD", "USDC", "USDT"):
                    sym = sym.replace("-", "/")
                converted.append((sym, chg, vol))
            top = converted

            if not top:
                logger.info("Momentum scan: no gainers found (market may be closed)")
                return

            # Log what we found
            for rank, (sym, chg, vol) in enumerate(top, 1):
                logger.info(
                    "  #%d %s: +%.2f%% (vol=%d)",
                    rank, sym, chg, vol,
                )

            # Extract just the symbols
            new_symbols = [s for s, _, _ in top]

            # Add to our tracked base symbols
            base_syms = set(self.symbols)
            all_syms = list(base_syms | set(new_symbols))
            self.symbols = all_syms

            # Subscribe dynamically
            self._add_dynamic_symbols(new_symbols)

            self._scanned_symbols = new_symbols
            self._last_scan_time = time.time()

            # Log sector breakdown
            sectors = self._momentum_scanner.get_trending_sectors(top)
            if sectors:
                logger.info("📊 MOMENTUM SECTORS: %s", sectors)

        except Exception as exc:
            logger.warning("Momentum scan failed: %s", exc)
            logger.debug(traceback.format_exc())

    # ------------------------------------------------------------------
    # WebSocket Handlers (Data Ingestion)
    # ------------------------------------------------------------------

    async def _on_bars(self, bar: Bar) -> None:
        """Process incoming bar data.

        This is the primary data pipeline trigger:
          bar -> update state -> compute indicators -> build features -> ML predict -> trade.
        """
        if not hasattr(bar, "symbol"):
            return
        try:
            sym = bar.symbol
            state = self.market_states.get(sym)
            if state is None:
                return

            state.update_price_buffers(bar)
            self.bar_count += 1

            # Compute technical indicators on the fly
            TechnicalIndicatorCalculator.compute_all(state)

            # Build the multi-modal feature vector and update ML
            if state.is_ready(self.trading_cfg.min_bars_for_indicators):
                try:
                    await self._process_features(state, bar)
                except Exception:
                    pass

                # Periodic ML retraining
                if self.bar_count % max(1, self.trading_cfg.retrain_interval_bars) == 0:
                    try:
                        self.ml_predictor.train()
                    except Exception:
                        pass

        except Exception as exc:
            logger.error("Error processing bar for %s: %s", bar.symbol if hasattr(bar, "symbol") else "?", exc)
            logger.debug(traceback.format_exc())

    async def _on_quotes(self, quote: Quote) -> None:
        """Process incoming quote data for order book imbalance tracking."""
        if not hasattr(quote, "symbol"):
            return
        try:
            state = self.market_states.get(quote.symbol)
            if state is not None:
                state.update_quote(quote)
        except Exception as exc:
            logger.debug("Error processing quote: %s", exc)

    async def _on_trades(self, trade: Any) -> None:
        """Process incoming trade data -> build synthetic bars (extended hours support).

        During after-hours/extended hours, Alpaca does not publish bar data.
        This handler aggregates individual trades into synthetic 1-minute bars
        so the ML pipeline can continue generating signals.
        """
        if not hasattr(trade, "symbol"):
            return
        try:
            sym = trade.symbol
            agg = self._trade_aggregators.get(sym)
            if agg is None:
                return
            if hasattr(trade, "price") and hasattr(trade, "size"):
                synthetic_bar = agg.add_trade(trade)
                if synthetic_bar is not None:
                    await self._on_bars(synthetic_bar)
        except Exception as exc:
            logger.debug("Trade processing error for %s: %s", sym if 'sym' in dir() else "?", exc)

    # ------------------------------------------------------------------
    # Feature Engineering & ML Pipeline
    # ------------------------------------------------------------------

    async def _process_features(self, state: MarketState, bar: Bar) -> None:
        """Build the unified multi-modal feature vector and update the ML model.

        Feature matrix construction (as described in the MLPredictor docstring):
          We merge text-derived sentiment features (FinBERT scores) with raw numerical
          market vectors (RSI, MACD, ATR, VWAP dev, order book imbalance, etc.) via
          horizontal concatenation. The resulting (n_samples, 16) matrix allows the
          downstream classifier to learn cross-modal patterns — e.g., bullish technical
          setups reinforced by positive news, or bearish divergences flagged by negative
          sentiment despite favorable price action.
        """
        # Update sentiment (throttled: once per bar cycle across all symbols)
        if self.bar_count % len(self.symbols) == 0:
            avg_sent, avg_conf = self.sentiment_engine.compute_aggregate_sentiment(
                self.symbols
            )
            for s in self.symbols:
                self.market_states[s].sentiment_score = avg_sent
                self.market_states[s].sentiment_confidence = avg_conf

        features = MLPredictor.build_feature_vector(state)

        # Determine label: 1 if next bar close > this bar close, else 0
        label = 1 if (len(state.closes) >= 2 and bar.close > state.closes[-2]) else 0

        self.ml_predictor.add_sample(features, label)

        # Run ML prediction
        pred_class, prob_up = self.ml_predictor.predict(features)
        state.ml_prediction = float(pred_class)
        state.ml_probability = prob_up

        # Optional LSTM update
        if self._enable_lstm and self._lstm_trainer is not None and _TORCH_AVAILABLE:
            self._lstm_update(features, label)

    def _lstm_update(self, features: np.ndarray, label: int) -> None:
        """Online LSTM training step (skeleton)."""
        try:
            if self._lstm_model is None:
                self._lstm_model = LSTMPredictorSkeleton(input_dim=len(features))
                self._lstm_trainer = LSTMTrainer(self._lstm_model)

            X = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            y = torch.tensor([label], dtype=torch.float32)
            loss = self._lstm_trainer.train_step(X, y)
            _ = loss  # logging not needed for skeleton
        except Exception as exc:
            logger.debug("LSTM update failed: %s", exc)

    # ------------------------------------------------------------------
    # Signal Generation & Trade Execution
    # ------------------------------------------------------------------

    async def _evaluate_signal(self, state: MarketState) -> None:
        """Combine ML prediction, technical scores, and sentiment into a trading signal.

        Two signal paths:
          A. MOMENTUM PATH (scanned gainers): trade on price action alone —
             if price > VWAP + volume spike + positive intraday momentum -> BUY.
             No ML required. Catches 'free risers' immediately.

          B. ML PATH (all symbols): full multi-modal pipeline requiring
             trained ML model. Used as secondary confirmation.

        Maximum aggression: momentum path always active, ML path adds
        confirmation when available.
        """
        # Skip signal evaluation when momentum monitor is handling trading
        if self.risk_manager.positions:
            return

        sym = state.symbol
        threshold = self.trading_cfg.prediction_threshold
        sentiment = state.sentiment_score

        # Determine if this is a momentum-scanned symbol (trade on price action)
        is_momentum_pick = sym in self._scanned_symbols

        direction = 0
        confidence = 0.0

        # --- PATH A: Pure Momentum ---
        # For scanned symbols without bar data, fetch price from yfinance directly
        has_bar_data = len(state.closes) >= self.trading_cfg.min_bars_for_indicators

        if is_momentum_pick or has_bar_data:
            if has_bar_data:
                # Stream-based momentum check
                vwap = state.vwap
                avg_vol = float(np.mean(state.volumes)) if len(state.volumes) >= 2 else 1.0
                last_vol = state.volumes[-1] if state.volumes else 1.0
                vol_ratio = last_vol / (avg_vol + 1e-8)
                above_vwap = state.current_price > vwap if vwap > 0 else True
                above_ema9 = state.current_price > state.ema_9 if state.ema_9 > 0 else True
                rsi_ok = 30 < state.rsi < 80

                if above_vwap and above_ema9 and vol_ratio > 0.8 and rsi_ok:
                    direction = 1
                    confidence = min(1.0, 0.5 + vol_ratio * 0.1 + (state.rsi - 50) / 100)
                    logger.debug(
                        "MOMENTUM SIGNAL: %s vwap=%.2f vol_ratio=%.1f rsi=%.1f → BUY (conf=%.2f)",
                        sym, vwap, vol_ratio, state.rsi, confidence,
                    )
            elif is_momentum_pick:
                # No stream data for this scanner pick — fetch price from yfinance
                # and trade on momentum score alone
                try:
                    loop = asyncio.get_running_loop()
                    ticker = await loop.run_in_executor(
                        None, lambda: yf.Ticker(sym).fast_info
                    )
                    price = float(getattr(ticker, 'last_price', 0) or getattr(ticker, 'regular_market_previous_close', 0) or 0)
                    if price > 0:
                        state.current_price = price
                        direction = 1  # momentum pick = BUY
                        confidence = 0.6  # moderate confidence, no stream data
                        logger.debug(
                            "MOMENTUM PICK (yfinance): %s @ $%.2f → BUY",
                            sym, price,
                        )
                except Exception as exc:
                    logger.debug("yfinance price fetch for %s failed: %s", sym, exc)

        # --- PATH B: ML Confirmation (if ML model is trained) ---
        if direction == 0 and self.ml_predictor._is_fitted:
            prob_up = state.ml_probability
            if prob_up >= threshold and sentiment >= -0.3:
                direction = 1
                confidence = prob_up
                logger.debug("ML SIGNAL: %s prob_up=%.3f → BUY", sym, prob_up)

        if direction == 0:
            return  # No actionable signal

        # Build signal
        tech_score = TechnicalIndicatorCalculator.ema_crossover_strength(state)
        signal = TradingSignal(
            symbol=sym,
            direction=direction,
            confidence=confidence,
            ml_probability=state.ml_probability,
            sentiment_score=sentiment,
            technical_score=tech_score,
            timestamp=datetime.now(timezone.utc),
            metadata={
                "atr": state.atr,
                "rsi": state.rsi,
                "order_book_imb": state.order_book_imbalance,
                "vol_ratio": float(np.mean(state.volumes)) if state.volumes else 0.0,
                "momentum_pick": is_momentum_pick,
                "dry_run": self.is_dry_run,
            },
        )

        await self._execute_trade(signal, state)

    async def _execute_trade(self, signal: TradingSignal, state: MarketState) -> None:
        """Validate risk constraints and submit the order to Alpaca.

        Steps:
          1. Fetch current account details (buying power, portfolio value).
          2. Run risk checks (ATR threshold, max exposure, existing position).
          3. Compute ATR-based position sizing.
          4. Determine stop loss / take profit levels.
          5. Submit order (or log it in dry-run mode).
        """
        if self._trading_client is None:
            logger.error("Trading client not initialized.")
            return

        # Skip all trades if any position is already open (momentum monitor handles this)
        if self.risk_manager.positions:
            return

        sym = signal.symbol
        side = OrderSide.BUY if signal.direction == 1 else OrderSide.SELL

        try:
            # --- Account Info ---
            account = self._trading_client.get_account()
            portfolio_value = float(account.equity) if not self.is_dry_run else self.trading_cfg.starting_cash
            buying_power = float(account.buying_power) if not self.is_dry_run else portfolio_value

            if portfolio_value <= 0:
                logger.warning("Portfolio value is %.2f; skipping trade.", portfolio_value)
                return

            # --- Risk Checks ---
            current_pos_value = sum(
                self.risk_manager.positions[p].quantity * state.current_price
                for p in self.risk_manager.positions
                if p != sym
            )

            allowed, reason = self.risk_manager.can_open_new_position(
                symbol=sym,
                portfolio_value=portfolio_value,
                position_value=current_pos_value + (portfolio_value * self.trading_cfg.max_position_pct),
                atr=state.atr,
                current_price=state.current_price,
            )
            if not allowed:
                logger.info("Risk check failed for %s: %s", sym, reason)
                return

            # --- Position Sizing (all-in momentum: 100% of available capital) ---
            notional = min(portfolio_value, buying_power, portfolio_value * 0.95)
            quantity = notional / state.current_price
            if quantity <= 0 or notional <= 0 or quantity < 0.0001:
                logger.info("Position size too small for %s (notional=%.2f); skipping.", sym, notional)
                return

            # --- Stop Loss (monitor handles all exits; trail-only, no TP) ---
            sl_price = state.current_price * (1 - self.hard_sl_pct / 100)

            # --- Order Execution ---
            if self.is_dry_run:
                logger.info(
                    "[DRY-RUN] %s $%.2f (%f shares) of %s @ %.2f | SL=%.2f | "
                    "ML_prob=%.3f Sentiment=%.3f",
                    "BUY" if side == OrderSide.BUY else "SELL",
                    notional, quantity, sym,
                    state.current_price,
                    sl_price,
                    signal.ml_probability,
                    signal.sentiment_score,
                )
                self.total_trades += 1

                open_pos = OpenPosition(
                    symbol=sym,
                    side=side,
                    entry_price=state.current_price,
                    quantity=float(quantity),
                    atr_at_entry=state.atr,
                    stop_loss_price=sl_price,
                    take_profit_price=0.0,
                    entry_time=datetime.now(timezone.utc),
                )
                self.risk_manager.positions[sym] = open_pos
                # Reset high/low watermarks for fresh trail tracking
                if hasattr(self, f'{sym}_high'):
                    delattr(self, f'{sym}_high')
                if hasattr(self, f'{sym}_low'):
                    delattr(self, f'{sym}_low')
                self._track_compounding()
                return

            # --- Live Order ---
            is_crypto = "/" in sym
            in_extended = False if is_crypto else is_extended_hours()

            if is_crypto:
                # Crypto: notional (spend USD), IOC, no brackets, no extended hours
                # Minimum crypto order is $10
                notional = min(notional, buying_power)
                if notional < 10.0:
                    logger.info("Crypto min $10 order not met for %s ($%.2f)", sym, notional)
                    return
                order_kwargs = dict(
                    symbol=sym,
                    notional=round(notional, 2),
                    side=side,
                    type=OrderType.MARKET,
                    time_in_force=TimeInForce.IOC,
                )
            else:
                # Stocks: notional, DAY, optional brackets, optional extended hours
                if in_extended:
                    limit_px = round(state.current_price * 1.005, 2)
                    order_kwargs = dict(
                        symbol=sym,
                        notional=round(notional, 2),
                        side=side,
                        type=OrderType.LIMIT,
                        limit_price=limit_px,
                        time_in_force=TimeInForce.DAY,
                        extended_hours=True,
                    )
                else:
                    order_kwargs = dict(
                        symbol=sym,
                        notional=round(notional, 2),
                        side=side,
                        type=OrderType.MARKET,
                        time_in_force=TimeInForce.DAY,
                    )
                # No Alpaca-side brackets — position monitor handles all exits

            order_req_class = LimitOrderRequest if (in_extended and not is_crypto) else MarketOrderRequest
            order_req = order_req_class(**order_kwargs)

            submitted = self._trading_client.submit_order(order_req)
            logger.info(
                "ORDER SUBMITTED: %s $%.2f %s @ market | OrderID=%s | SL=%.2f (trail-only, no TP)",
                side.value, notional, sym,
                submitted.id, sl_price,
            )
            self.total_trades += 1

            # Track position immediately on submission (prevents double-trading)
            open_pos = OpenPosition(
                symbol=sym,
                side=side,
                entry_price=state.current_price,
                quantity=float(quantity),
                atr_at_entry=state.atr,
                stop_loss_price=sl_price,
                take_profit_price=0.0,
                entry_time=datetime.now(timezone.utc),
            )
            self.risk_manager.positions[sym] = open_pos
            # Reset high/low watermarks for fresh trail tracking
            for attr in [f'{sym}_high', f'{sym}_low', f'{sym}_yf_ts', f'{sym}_price_ts']:
                if hasattr(self, attr):
                    delattr(self, attr)
            if submitted.status != OrderStatus.FILLED:
                logger.info("Order pending — position tracked for %s", sym)
            self._track_compounding()

        except APIError as exc:
            logger.error("Alpaca API error for %s: %s", sym, exc)
        except Exception as exc:
            logger.error("Trade execution failed for %s: %s", sym, exc)
            logger.debug(traceback.format_exc())

    # ------------------------------------------------------------------
    # Position Monitoring & Risk Management
    # ------------------------------------------------------------------

    async def _monitor_positions(self) -> None:
        """Position monitor — cycles every 2 seconds.

        Exits (no TP — trail-only):
          - HARD STOP: hard_sl_pct% below entry
          - TRAILING STOP: activates at trail_act_pct% gain, trail_dist_pct% trail
        """
        log_counter = 0
        while not self._shutdown_event.is_set():
            try:
                for sym in list(self.risk_manager.positions.keys()):
                    pos = self.risk_manager.positions[sym]
                    state = self.market_states.get(sym)
                    if state is None or state.current_price == 0:
                        continue

                    # Refresh price: Alpaca every 5s, yfinance fallback if stale
                    now = time.time()
                    last_refresh = getattr(self, f'_price_ts_{sym}', 0)
                    if now - last_refresh >= 5:
                        alpaca_price = 0.0
                        try:
                            ap = self._trading_client.get_position(sym)
                            alpaca_price = float(getattr(ap, 'current_price', 0) or 0)
                        except Exception:
                            pass

                        price = alpaca_price
                        # If Alpaca price is stale (equals entry), use yfinance
                        if price <= 0 or (price == pos.entry_price and self.market_states.get(sym) and
                                          self.market_states[sym].current_price == pos.entry_price):
                            yf_refresh = getattr(self, f'_yf_ts_{sym}', 0)
                            if now - yf_refresh >= 30:
                                try:
                                    import yfinance as yf
                                    info = yf.Ticker(sym).fast_info
                                    live = float(getattr(info, 'last_price', 0) or 0)
                                    if live > 0 and live != pos.entry_price:
                                        price = live
                                        setattr(self, f'_yf_ts_{sym}', now)
                                except Exception:
                                    pass

                        if price > 0:
                            state.current_price = price
                            setattr(self, f'_price_ts_{sym}', now)

                    p = state.current_price
                    entry = pos.entry_price

                    # Log status every ~60s
                    log_counter += 1
                    if log_counter % 30 == 0:
                        gain_pct = ((p - entry) / entry) * 100
                        high_water = getattr(self, f'{sym}_high', entry)
                        trail = high_water * (1 - self.trail_dist_pct / 100) if high_water >= entry * (1 + self.trail_act_pct / 100) else 0
                        logger.info(
                            "📊 %s: $%.2f (entry=$%.2f, gain=%.2f%%, high=$%.2f, trail=$%.2f)",
                            sym, p, entry, gain_pct, high_water, trail,
                        )

                    # 1) HARD STOP
                    hard_sl = entry * (1 - self.hard_sl_pct / 100)
                    if not pos.is_long:
                        hard_sl = entry * (1 + self.hard_sl_pct / 100)

                    if (pos.is_long and p <= hard_sl) or (not pos.is_long and p >= hard_sl):
                        logger.warning(
                            "🛑 HARD STOP: %s at %.2f (entry=%.2f, loss=%.2f%%)",
                            sym, p, entry, ((p - entry) / entry) * 100,
                        )
                        await self._close_position(sym, reason="stop_loss")
                        continue

                    # 2) STALE EXIT: sell if price hasn't moved in 5 min
                    held_seconds = (datetime.now(timezone.utc) - pos.entry_time).total_seconds()
                    gain_pct_check = ((p - entry) / entry) * 100 if entry else 0
                    if held_seconds > 600 and abs(gain_pct_check) < self.trail_act_pct / 2:
                        logger.info(
                            "⏰ STALE EXIT: %s held %.0fs, gain=%.2f%% — selling to free cash for next pick",
                            sym, held_seconds, gain_pct_check,
                        )
                        await self._close_position(sym, reason="stale_exit")
                        continue

                    # 3) TRAILING STOP — no TP cap, trail lets winners run
                    high_key = f"{sym}_high"
                    low_key = f"{sym}_low"
                    if pos.is_long:
                        prev_high = getattr(self, high_key, entry)
                        new_high = max(prev_high, p)
                        setattr(self, high_key, new_high)

                        if new_high >= entry * (1 + self.trail_act_pct / 100):
                            trail_stop = new_high * (1 - self.trail_dist_pct / 100)
                            if p <= trail_stop:
                                logger.info(
                                    "🎯 TRAIL HIT: %s at %.2f (high=%.2f, trail=%.2f, gain=%.2f%%)",
                                    sym, p, new_high, trail_stop, ((p - entry) / entry) * 100,
                                )
                                await self._close_position(sym, reason="trailing_stop")
                                continue
                    else:
                        prev_low = getattr(self, low_key, entry)
                        new_low = min(prev_low, p)
                        setattr(self, low_key, new_low)

                        if new_low <= entry * (1 - self.trail_act_pct / 100):
                            trail_stop = new_low * (1 + self.trail_dist_pct / 100)
                            if p >= trail_stop:
                                await self._close_position(sym, reason="trailing_stop")
                                continue

                    # NO TAKE PROFIT — trail-only lets winners run

            except Exception as exc:
                logger.debug("Position monitor error: %s", exc)

            await asyncio.sleep(2)

    async def _close_position(self, symbol: str, reason: str = "manual") -> None:
        """Close a position via Alpaca API — uses close_position() which handles qty correctly."""
        pos = self.risk_manager.positions.pop(symbol, None)
        if pos is None:
            return

        if self.is_dry_run:
            logger.info("[DRY-RUN] CLOSE %s %.6f shares", symbol, pos.quantity)
            return

        # Record trade outcome for progressive learning
        try:
            exit_px = self.market_states.get(symbol, MarketState(symbol)).current_price
            if exit_px <= 0:
                exit_px = pos.entry_price
            profit_pct = pos.current_pnl_pct(exit_px)
            record = TradeRecord(
                symbol=symbol,
                entry_price=pos.entry_price,
                exit_price=exit_px,
                profit_pct=profit_pct * 100,
                entry_time=pos.entry_time,
                exit_time=datetime.now(timezone.utc),
                reason=reason,
            )
            self._ledger.record(record)
            logger.info("📋 LEDGER: %s %+.2f%% (%s)", symbol, profit_pct * 100, reason)
        except Exception as exc:
            logger.debug("Ledger record error: %s", exc)

        try:
            self._trading_client.close_position(symbol)
            logger.info("Position closed: %s", symbol)
        except Exception:
            try:
                is_crypto = "/" in symbol
                side = OrderSide.SELL if pos.is_long else OrderSide.BUY
                ap = self._trading_client.get_position(symbol)
                actual_qty = abs(float(ap.qty))
                clean_qty = actual_qty if is_crypto else math.floor(actual_qty * 1e6) / 1e6
                if clean_qty < 0.0001:
                    return
                self._trading_client.submit_order(MarketOrderRequest(
                    symbol=symbol, qty=clean_qty, side=side,
                    type=OrderType.MARKET,
                    time_in_force=TimeInForce.DAY,
                    extended_hours=not is_crypto and is_extended_hours(),
                ))
                logger.info("Position closed (fallback): %s %s (qty=%.6f)", symbol, side.value, clean_qty)
            except Exception as exc:
                logger.error("Failed to close position %s: %s", symbol, exc)

        if not hasattr(self, '_sold_cooldowns'):
            self._sold_cooldowns = {}
        self._sold_cooldowns[symbol] = time.time()

        # Also mark stale/stop exits with extended cooldown to prevent re-buy loops
        if reason in ("stale_exit", "stop_loss"):
            self._sold_cooldowns[f"hard_{symbol}"] = time.time()

    # ------------------------------------------------------------------
    # Compounding Performance Tracker
    # ------------------------------------------------------------------

    def _track_compounding(self) -> None:
        """Log portfolio growth and progress toward the $1B target after each trade.

        This tracks the exponential compounding curve. Each trade that closes
        successfully adds to the compounding base. The engine aims to
        achieve 0.5-2% per trade with max aggression settings.
        """
        try:
            if self._trading_client and not self.is_dry_run:
                account = self._trading_client.get_account()
                equity = float(account.equity)
            else:
                equity = self.trading_cfg.starting_cash

            if equity > self._peak_equity:
                self._peak_equity = equity
                self._compounding_periods += 1

            growth = equity - self._last_logged_equity
            if abs(growth) > 0.01:
                self._total_return_pct = ((equity / self.trading_cfg.starting_cash) - 1) * 100
                pct_to_target = (equity / self._target_equity) * 100
                multiples = equity / self.trading_cfg.starting_cash

                logger.info(
                    "💰 COMPOUND: $%.2f | +$%.2f | +%.4f%% total | %.2fx return | "
                    "%.8f%% of $1B goal | %d trades",
                    equity,
                    growth,
                    self._total_return_pct,
                    multiples,
                    pct_to_target,
                    self.total_trades,
                )
                self._last_logged_equity = equity

        except Exception as exc:
            logger.debug("Compounding track error: %s", exc)

    # ------------------------------------------------------------------
    # Account Summary
    # ------------------------------------------------------------------

    async def _log_account_summary(self) -> None:
        """Periodically log account status and open positions."""
        while not self._shutdown_event.is_set():
            try:
                if self._trading_client and not self.is_dry_run:
                    account = self._trading_client.get_account()
                    equity = float(account.equity)
                    pct = (equity / self._target_equity) * 100
                    mult = equity / self.trading_cfg.starting_cash
                    logger.info(
                        "🚀 ROAD TO $1B: $%.2f (%.2fx, %.6f%% of target) | "
                        "BP=%.2f | Trades=%d | Bars=%d | Open=%d",
                        equity, mult, pct,
                        float(account.buying_power),
                        self.total_trades,
                        self.bar_count,
                        len(self.risk_manager.positions),
                    )
                    self._last_logged_equity = equity

                    # Log trade ledger summary every 5th cycle
                    if self.total_trades > 0 and self.bar_count % 5 == 0:
                        logger.debug(self._ledger.summary())
                else:
                    logger.info(
                        "[DRY-RUN] Portfolio=%.2f, OpenPositions=%d, "
                        "TotalTrades=%d, BarsProcessed=%d",
                        self.trading_cfg.starting_cash,
                        len(self.risk_manager.positions),
                        self.total_trades,
                        self.bar_count,
                    )

                # Log feature importance periodically
                importance = self.ml_predictor.get_feature_importance()
                if importance:
                    top_features = dict(sorted(importance.items(), key=lambda x: -x[1])[:5])
                    logger.debug("Top ML features: %s", top_features)

            except Exception as exc:
                logger.debug("Account summary error: %s", exc)

            await asyncio.sleep(60)

    # ------------------------------------------------------------------
    # Lazy LSTM Enablement
    # ------------------------------------------------------------------

    def enable_lstm(self) -> None:
        """Enable the optional PyTorch LSTM predictor path."""
        if not _TORCH_AVAILABLE:
            logger.warning("PyTorch not available; LSTM cannot be enabled.")
            return
        self._enable_lstm = True
        logger.info("LSTM predictor path enabled.")

    async def _periodic_momentum_scan(self) -> None:
        """Periodically scan for momentum gainers throughout the trading day."""
        while not self._shutdown_event.is_set():
            now = time.time()
            if now - self._last_scan_time >= self._scan_interval:
                await self._run_momentum_scan()
            await asyncio.sleep(60)

    async def _monitor_momentum_picks(self) -> None:
        """MAX AGGRESSION MOMENTUM MONITOR — runs every 5s, 24/7.

        Strategy:
          1. Scan the FULL universe via yfinance batch download
          2. Pick the #1-#3 gainer with > 0.3% daily change
          3. Go ALL IN — 100% of available capital
          4. No filters, no sleep, no risk limits
        """
        while not self._shutdown_event.is_set():
            try:
                if self.risk_manager.positions:
                    await asyncio.sleep(3)
                    continue

                await self._cancel_all_unfilled()

                if self._trading_client and not self.is_dry_run:
                    try:
                        account = self._trading_client.get_account()
                        cash = float(account.cash)
                        equity = float(account.equity)
                    except Exception:
                        cash = 0.0
                        equity = 0.0
                else:
                    cash = self.trading_cfg.starting_cash
                    equity = cash

                # Check for UNTRACKED positions on Alpaca (not in risk_manager)
                if cash < equity * 0.5 and equity >= 5.0:
                    try:
                        alpaca_positions = self._trading_client.get_all_positions()
                        for ap in alpaca_positions:
                            ap_sym = ap.symbol
                            ap_qty = abs(float(ap.qty))
                            if ap_qty < 0.0001:
                                continue
                            if ap_sym in self.risk_manager.positions:
                                continue
                            side = OrderSide.SELL if float(ap.qty) > 0 else OrderSide.BUY
                            is_crypto = "/" in ap_sym
                            clean_qty = math.floor(ap_qty * 1e6) / 1e6 if not is_crypto else ap_qty
                            if is_extended_hours() and not is_crypto:
                                price = round(float(ap.current_price) * 0.97, 2)
                                self._trading_client.submit_order(LimitOrderRequest(
                                    symbol=ap_sym, qty=clean_qty, side=side,
                                    limit_price=str(price),
                                    time_in_force=TimeInForce.DAY, extended_hours=True))
                            else:
                                self._trading_client.submit_order(MarketOrderRequest(
                                    symbol=ap_sym, qty=clean_qty, side=side,
                                    type=OrderType.MARKET,
                                    time_in_force=TimeInForce.IOC if is_crypto else TimeInForce.DAY,
                                    extended_hours=not is_crypto and is_extended_hours()))
                            logger.info("Selling untracked position: %s (qty=%.6f)", ap_sym, clean_qty)
                            if not hasattr(self, '_sold_cooldowns'):
                                self._sold_cooldowns = {}
                            self._sold_cooldowns[ap_sym] = time.time()
                            await asyncio.sleep(1)
                    except Exception as exc:
                        logger.debug("Untracked position cleanup: %s", exc)
                    await asyncio.sleep(5)
                    continue

                if cash < 3.0:
                    await asyncio.sleep(5)
                    continue

                # After-hours check: skip new stock entries outside regular hours
                if not self._crypto_mode:
                    now_et = _us_eastern_now()
                    tod = _time_to_tod(now_et)
                    is_weekday = now_et.weekday() < 5
                    is_open = is_weekday and REGULAR_START <= tod <= REGULAR_END
                    if not is_open:
                        await asyncio.sleep(10)
                        continue

                # Hot universe for fast scan
                if self._crypto_mode:
                    hot_list = self._crypto_hot_symbols
                else:
                    if not hasattr(self, '_hot_symbols'):
                        self._hot_symbols = [
                            "ARM","NVDA","AMD","MU","PLTR","HOOD","COIN","MSTR",
                            "TSLA","MARA","RIOT","SOUN","IONQ","RGTI","DKNG",
                            "CHWY","CVNA","SOFI","UPST","AFRM","GME","AMC",
                            "RIVN","LCID","TQQQ","SOXL","FAS","LABU","UPRO",
                            "FNGU","SPXL","NVDL","BITO","IBIT","CLSK",
                            "SMCI","CELH","DASH","UBER","SNOW","DDOG","NET",
                            "CRWD","PANW","ZS","MDB","AI","BBAI",
                        ]
                    hot_list = self._hot_symbols

                # --- SCAN: hot symbols first (fast), full universe fallback ---
                best_sym = None
                best_change = 0.0
                best_price = 0.0
                is_extended = True

                if not hasattr(self, '_scan_backoff'):
                    self._scan_backoff = 1

                try:
                    if self._scan_backoff <= 3:
                        top = await self._momentum_scanner.scan_via_yfinance(n=5, symbols=hot_list)
                        if not top:
                            if self._crypto_mode:
                                top = await self._momentum_scanner.scan_via_yfinance(n=5, symbols=self._crypto_all_symbols)
                            else:
                                top = await self._momentum_scanner.scan_via_yfinance(n=5)
                    else:
                        if self._crypto_mode:
                            top = await self._momentum_scanner.scan_via_yfinance(n=5, symbols=self._crypto_all_symbols)
                        else:
                            top = await self._momentum_scanner.scan_via_yfinance(n=5)
                    self._scan_backoff = max(1, self._scan_backoff - 1)

                    # Try top 5 candidates in order, pick first that passes
                    for idx, (sym, chg, _, close_px, prev_close) in enumerate(top):
                        if prev_close <= 0:
                            continue
                        # Skip if sold within last 5 min (10 min for stale/stop exits)
                        alpaca_sym = sym.replace("-USD", "/USD") if "-USD" in sym else sym
                        if hasattr(self, '_sold_cooldowns') and alpaca_sym in self._sold_cooldowns:
                            if time.time() - self._sold_cooldowns[alpaca_sym] < 300:
                                continue
                        # Hard cooldown for stale/stop exits — prevents re-buy loops
                        hard_key = f"hard_{alpaca_sym}"
                        if hasattr(self, '_sold_cooldowns') and hard_key in self._sold_cooldowns:
                            if time.time() - self._sold_cooldowns[hard_key] < 600:
                                continue
                        # Skip if same change% as last scan (stale — not moving)
                        last_pct = getattr(self, f'_scan_pct_{alpaca_sym}', None)
                        if last_pct is not None and abs(chg - last_pct) < 0.5:
                            continue
                        setattr(self, f'_scan_pct_{alpaca_sym}', chg)
                        try:
                            info = yf.Ticker(sym).fast_info
                            live = float(getattr(info, 'last_price', 0) or 0)
                            if live <= 0:
                                continue
                            # Trend filter: skip if below 50-day SMA (downtrend)
                            sma50 = getattr(info, 'fifty_day_average', None)
                            if sma50 and sma50 > 0 and live < sma50 * 1.003:
                                logger.debug("SKIP %s: below 50-SMA (%.2f < %.2f)", sym, live, sma50)
                                continue
                            # Volume filter: skip if volume is weak (no conviction)
                            avg_vol = getattr(info, 'average_volume', None)
                            curr_vol = getattr(info, 'regular_market_volume', None)
                            if avg_vol and curr_vol and avg_vol > 0 and curr_vol < avg_vol * 0.8:
                                logger.debug("SKIP %s: low volume (%.0f < %.0f avg)", sym, curr_vol, avg_vol)
                                continue
                        except Exception:
                            continue
                        live_chg = (live - prev_close) / max(prev_close, 0.01) * 100
                        if 0.5 <= live_chg <= 20.0 and not math.isnan(float(live_chg)):
                            best_sym = sym
                            best_change = live_chg
                            best_price = live
                            logger.info("📡 PICK #%d: %s +%.2f%%", idx + 1, sym, live_chg)
                            break
                except Exception as exc:
                    self._scan_backoff = min(10, self._scan_backoff + 2)
                    logger.debug("Scan failed: %s", exc)

                if best_sym is None or best_price <= 0:
                    await asyncio.sleep(5)
                    continue

                # Convert yfinance format to Alpaca format for trading
                if "-USD" in best_sym:
                    best_sym = best_sym.replace("-USD", "/USD")

                state = self.market_states.get(best_sym)
                if state is None:
                    state = MarketState(symbol=best_sym)
                    self.market_states[best_sym] = state
                state.current_price = best_price

                logger.info(
                    "🔥 ALL-IN MOMENTUM: %s @ $%.2f (+%.2f%%) | Cash=$%.2f",
                    best_sym, best_price, best_change, cash,
                )

                # ---- ML CONFIRMATION ----
                if self.ml_predictor._is_fitted:
                    try:
                        loop = asyncio.get_running_loop()
                        yf_sym = best_sym.replace("/USD", "-USD")
                        hist = await loop.run_in_executor(
                            None, lambda: yf.Ticker(yf_sym).history(period="2mo")
                        )
                        if hist is not None and len(hist) >= 20:
                            state.closes.clear()
                            state.highs.clear()
                            state.lows.clear()
                            state.volumes.clear()
                            for _, row in hist.iterrows():
                                state.closes.append(float(row['Close']))
                                state.highs.append(float(row['High']))
                                state.lows.append(float(row['Low']))
                                state.volumes.append(float(row['Volume']))
                            TechnicalIndicatorCalculator.compute_all(state)
                            features = MLPredictor.build_feature_vector(state)
                            _, prob_up = self.ml_predictor.predict(features)
                            state.ml_probability = prob_up
                            logger.info("🤖 ML: %s prob_up=%.1f%%", best_sym, prob_up * 100)
                            if prob_up < 0.55:
                                logger.info("❌ ML rejected %s (prob_up=%.1f%% < 55%%)", best_sym, prob_up * 100)
                                await asyncio.sleep(30)
                                continue
                    except Exception as exc:
                        logger.debug("ML check failed for %s: %s", best_sym, exc)

                signal = TradingSignal(
                    symbol=best_sym,
                    direction=1,
                    confidence=min(0.95, 0.3 + best_change / 50),
                    ml_probability=0.5,
                    sentiment_score=0.0,
                    technical_score=best_change / 100,
                    timestamp=datetime.now(timezone.utc),
                    metadata={
                        "all_in_momentum": True,
                        "yfinance_change_pct": best_change,
                        "dry_run": self.is_dry_run,
                    },
                )
                await self._execute_trade(signal, state)

            except Exception as exc:
                logger.debug("Momentum monitor error: %s", exc)

            await asyncio.sleep(30)

    async def _cancel_all_unfilled(self) -> None:
        """Cancel all open (unfilled) orders to free up buying power."""
        if self._trading_client is None or self.is_dry_run:
            return
        try:
            self._trading_client.cancel_orders()
        except Exception as exc:
            logger.debug("Cancel orders error: %s", exc)

    # ------------------------------------------------------------------
    # Stream Management (24/7 operation)
    # ------------------------------------------------------------------

    async def _run_stream_with_reconnect(self, stream: Any, name: str) -> None:
        """Run a data stream with infinite reconnection (for 24/7 operation).

        Stocks stream: runs during market hours
        Crypto stream: runs 24/7/365
        """
        reconnect_attempts = 0
        max_reconnects = self.sys_cfg.max_reconnect_attempts

        while not self._shutdown_event.is_set() and reconnect_attempts < max_reconnects:
            try:
                logger.info("[%s] Connecting (attempt %d)...", name, reconnect_attempts + 1)
                await stream._run_forever()
                # If _run_forever exits cleanly, it was stopped intentionally
                break
            except asyncio.CancelledError:
                break
            except Exception as exc:
                reconnect_attempts += 1
                delay = min(
                    self._reconnect_delay * (2 ** (reconnect_attempts - 1)),
                    60,
                )
                logger.warning(
                    "[%s] Disconnected: %s. Reconnecting in %.1fs (attempt %d/%d)...",
                    name, exc, delay, reconnect_attempts, max_reconnects,
                )
                await asyncio.sleep(delay)

        if reconnect_attempts >= max_reconnects:
            logger.critical("[%s] Max reconnect attempts reached.", name)

    async def _clean_existing_positions(self) -> None:
        """Close all existing positions and cancel pending orders on startup.

        Prevents stale positions from causing double-trades on restart.
        """
        if self.is_dry_run or self._trading_client is None:
            return
        try:
            positions = self._trading_client.get_all_positions()
            for p in positions:
                qty = abs(float(p.qty))
                if qty < 0.0001:
                    continue
                side = OrderSide.SELL if float(p.qty) > 0 else OrderSide.BUY
                is_crypto = "/" in p.symbol
                clean_qty = math.floor(qty * 1e6) / 1e6 if not is_crypto else qty
                in_extended = is_extended_hours()
                if in_extended and not is_crypto:
                    price = round(float(p.current_price) * 0.98, 2)
                    self._trading_client.submit_order(LimitOrderRequest(
                        symbol=p.symbol, qty=clean_qty, side=side,
                        limit_price=str(price),
                        time_in_force=TimeInForce.DAY, extended_hours=True,
                    ))
                else:
                    self._trading_client.submit_order(MarketOrderRequest(
                        symbol=p.symbol, qty=clean_qty, side=side,
                        type=OrderType.MARKET,
                        time_in_force=TimeInForce.IOC if is_crypto else TimeInForce.DAY,
                        extended_hours=in_extended and not is_crypto,
                    ))
                logger.info("Cleaned up existing position: %s", p.symbol)

            # Cancel any pending orders
            try:
                self._trading_client.cancel_orders()
            except Exception:
                for o in self._trading_client.get_orders():
                    if o.status not in ("filled", "cancelled", "done_for_day"):
                        try:
                            self._trading_client.cancel_orders()
                        except Exception:
                            pass
        except Exception as exc:
            logger.warning("Position cleanup error (non-fatal): %s", exc)

    async def _seed_historical_data(self) -> None:
        """Seed the ML model with recent historical bar data for instant readiness.

        Fetches the last 200 bars for each symbol via REST API and feeds them
        through the feature pipeline so the model has training data immediately.
        """
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame

            if not self.symbols:
                return

            hc = StockHistoricalDataClient(
                self.alpaca_cfg.api_key, self.alpaca_cfg.secret_key
            )
            now = datetime.now(timezone.utc)
            req = StockBarsRequest(
                symbol_or_symbols=self.symbols,
                timeframe=TimeFrame.Minute,
                start=now - timedelta(days=7),
                end=now,
                limit=200,
            )
            bars = hc.get_stock_bars(req)

            seeded = 0
            for sym in self.symbols:
                sym_bars = bars.data.get(sym, [])
                if not sym_bars:
                    continue
                state = self.market_states.get(sym)
                if state is None:
                    continue
                for bar in sym_bars:
                    state.update_price_buffers(bar)
                    TechnicalIndicatorCalculator.compute_all(state)
                    if state.is_ready(10):
                        f = MLPredictor.build_feature_vector(state)
                        label = 1 if len(state.closes) >= 2 and bar.close > state.closes[-2] else 0
                        self.ml_predictor.add_sample(f, label)
                        seeded += 1

            if seeded > 0:
                trained = self.ml_predictor.train()
                logger.info(
                    "📊 HISTORICAL SEEDING: %d samples loaded, model trained=%s",
                    seeded, trained,
                )
            else:
                logger.info("Historical seeding: no recent bars available (after-hours expected)")
        except Exception as exc:
            logger.debug("Historical seeding error (non-fatal): %s", exc)

    async def _weekend_prep(self) -> None:
        """Weekend preparation: warm data, train models, cache everything.

        Downloads 60 days of data for ALL universe symbols, warms MarketState
        objects with indicators, seeds ML models, and auto-tunes exit params.
        Only runs during weekends or when no positions are open.
        """
        try:
            now_et = _us_eastern_now()
            is_weekend = now_et.weekday() >= 5
            has_positions = bool(self.risk_manager.positions)

            if has_positions:
                logger.info("Prep skipped: positions open")
                return

            logger.info("=" * 60)
            logger.info("WEEKEND PREP: warming data, training models, tuning params")
            logger.info("=" * 60)

            # 1) Download daily data for entire universe
            import yfinance as yf
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler

            if self._crypto_mode:
                universe = self._crypto_all_symbols
            else:
                universe = self._momentum_scanner.UNIVERSE if hasattr(self._momentum_scanner, 'UNIVERSE') else self.symbols
            logger.info("Downloading %s data for %d symbols ...", self.ml_data_period, len(universe))

            all_data = {}
            batch_size = 20
            for i in range(0, len(universe), batch_size):
                batch = universe[i:i + batch_size]
                try:
                    data = yf.download(batch, period=self.ml_data_period, progress=False, auto_adjust=True, group_by="ticker")
                    for sym in batch:
                        try:
                            sd = data[sym] if isinstance(data.columns, pd.MultiIndex) and sym in data else data
                            if sd.empty or "Close" not in sd.columns:
                                continue
                            df = pd.DataFrame({
                                "close": sd["Close"], "high": sd["High"],
                                "low": sd["Low"], "volume": sd["Volume"],
                            }).dropna()
                            if len(df) >= 20:
                                all_data[sym] = df
                        except Exception:
                            continue
                except Exception as e:
                    logger.debug("Batch download error: %s", e)
                await asyncio.sleep(0.5)

            logger.info("Downloaded %d symbols", len(all_data))

            # 2) Warm MarketState for each symbol with full indicators
            warmed = 0
            ml_samples = 0
            all_features = []
            all_labels = []

            for sym, df in all_data.items():
                state = self.market_states.get(sym)
                if state is None:
                    state = MarketState(symbol=sym)
                    self.market_states[sym] = state

                closes = df["close"].values
                highs = df["high"].values
                lows = df["low"].values
                volumes = df["volume"].values

                state.current_price = float(closes[-1]) if len(closes) > 0 else 0

                for j in range(len(closes)):
                    state.closes.append(float(closes[j]))
                    state.highs.append(float(highs[j]))
                    state.lows.append(float(lows[j]))
                    state.volumes.append(float(volumes[j]))

                    if len(state.closes) >= 10:
                        TechnicalIndicatorCalculator.compute_all(state)
                        fv = MLPredictor.build_feature_vector(state)
                        all_features.append(fv)
                        label = 1 if j >= 1 and closes[j] > closes[j-1] else 0
                        all_labels.append(label)

                if len(closes) >= 10:
                    warmed += 1

            logger.info("Warmed %d MarketState objects", warmed)

            # 3) Seed ML model with full feature matrix
            if len(all_features) >= 100:
                X = np.vstack(all_features)
                y = np.array(all_labels)

                for fv, lbl in zip(X, y):
                    self.ml_predictor.add_sample(fv, lbl)

                trained = self.ml_predictor.train()
                logger.info("ML seeded: %d samples, trained=%s", len(all_features), trained)

                # Also train a standalone model for persistence
                os.makedirs("models", exist_ok=True)
                scaler = StandardScaler()
                Xs = scaler.fit_transform(X)
                rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
                rf.fit(Xs, y)
                try:
                    import joblib
                    joblib.dump(rf, "models/rf_model.joblib")
                    joblib.dump(scaler, "models/scaler.joblib")
                    logger.info("Models saved to models/")
                except Exception:
                    pass

            # 4) Quick parameter sweep to auto-tune exits
            logger.info("Auto-tuning exit params via backtest sweep ...")
            best_ret = 0
            best_params = {"tp": 0.6, "sl": 0.4, "trail_act": 0.3, "trail_dist": 0.3}

            common_dates = sorted(set.intersection(
                *[set(df.index) for df in all_data.values() if len(df) > 0]
            ))
            if len(common_dates) >= 20:
                for tp_pct in [0.4, 0.6, 0.8, 1.0]:
                    for sl_pct in [0.3, 0.4, 0.5, 0.6]:
                        if sl_pct >= tp_pct:
                            continue
                        # Simulate quickly on first 500 dates
                        cap = 100.0
                        pos = None
                        trades = 0
                        wins = 0
                        for dt in common_dates[:500]:
                            if pos is not None:
                                sym, entry, shares = pos
                                if sym in all_data and dt in all_data[sym].index:
                                    day = all_data[sym].loc[dt]
                                    hi, lo = float(day["high"]), float(day["low"])
                                    hard_sl = entry * (1 - sl_pct / 100)
                                    tp = entry * (1 + tp_pct / 100)
                                    if lo <= hard_sl:
                                        cap += shares * hard_sl
                                        pos = None
                                    elif hi >= tp:
                                        cap += shares * tp
                                        trades += 1
                                        wins += 1
                                        pos = None
                            if pos is not None:
                                continue
                            candidates = []
                            for sym2, df2 in all_data.items():
                                if dt not in df2.index:
                                    continue
                                idx = list(df2.index).index(dt)
                                if idx == 0:
                                    continue
                                pc = float(df2.iloc[idx - 1]["close"])
                                cc = float(df2.iloc[idx]["close"])
                                if pc <= 0:
                                    continue
                                chg = (cc - pc) / pc * 100
                                if 1.0 <= chg <= 12.0:
                                    candidates.append((sym2, chg, cc))
                            if candidates:
                                candidates.sort(key=lambda x: x[1], reverse=True)
                                sym2, chg, price = candidates[0]
                                shares = cap / price
                                cap = 0
                                pos = (sym2, price, shares)

                total_ret = (cap - 100) / 100 * 100
                if total_ret > best_ret:
                    best_ret = total_ret
                    best_params = {"tp": tp_pct, "sl": sl_pct, "trail_act": 0.3, "trail_dist": 0.3}

                logger.info("Auto-tune complete: best TP=%.1f%% SL=%.1f%% (return=%.1f%%)",
                            best_params["tp"], best_params["sl"], best_ret)

            # 5) Cache the downloaded data for instant first scan on Monday
            try:
                import yfinance as yf
                cached = {"close": {}, "prev_close": {}}
                for sym, df in all_data.items():
                    if len(df) > 1:
                        cached["close"][sym] = float(df["close"].iloc[-1])
                        cached["prev_close"][sym] = float(df["close"].iloc[-2])
                pd.to_pickle(cached, "models/price_cache.pkl")
                logger.info("Price cache saved to models/price_cache.pkl")
            except Exception as exc:
                logger.debug("Cache save error: %s", exc)

            logger.info("=" * 60)
            logger.info("WEEKEND PREP COMPLETE")
            logger.info("=" * 60)

        except Exception as exc:
            logger.debug("Weekend prep error (non-fatal): %s", exc)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def shutdown(self, sig: Optional[int] = None) -> None:
        """Graceful shutdown: stop streams, cancel tasks, close positions."""
        if sig:
            logger.info("Received signal %d; initiating graceful shutdown...", sig)

        self.is_running = False
        self._shutdown_event.set()

        # Cancel all background tasks
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        # Close WebSockets
        for s in [getattr(self, '_stock_stream', None), getattr(self, '_stream', None)]:
            if s is not None:
                try:
                    s.stop()
                    await s.close()
                except Exception as exc:
                    logger.debug("Error closing stream: %s", exc)

        # Close any open positions (optional, depending on strategy)
        if self.risk_manager.positions:
            logger.warning(
                "Shutdown with %d open positions. Consider manual resolution.",
                len(self.risk_manager.positions),
            )

        logger.info(
            "MultiModalTradingEngine shutdown. Total trades: %d, Bars: %d",
            self.total_trades,
            self.bar_count,
        )

    # ------------------------------------------------------------------
    # Main Execution Loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Main async entry point for the trading engine."""
        if self.is_running:
            logger.warning("Engine is already running.")
            return

        self.is_running = True
        self.start_time = datetime.now(timezone.utc)

        logger.info("=" * 60)
        logger.info("MultiModalTradingEngine STARTING")
        logger.info("  Symbols:         %s", self.symbols)
        logger.info("  Dry-run:         %s", self.is_dry_run)
        logger.info("  Paper trading:   %s", self.is_paper)
        logger.info("  Starting cash:   %.2f", self.trading_cfg.starting_cash)
        logger.info("  Max pos %%:       %.1f%%", self.trading_cfg.max_position_pct * 100)
        logger.info("  Stop-loss mult:  %.1f ATR", self.trading_cfg.atr_stop_loss_multiplier)
        logger.info("  Take-profit mult:%.1f ATR", self.trading_cfg.atr_take_profit_multiplier)
        logger.info("  Prediction thresh: %.2f", self.trading_cfg.prediction_threshold)
        logger.info("  FinBERT model:   %s", self.sentiment_cfg.model_name)
        logger.info("=" * 60)

        try:
            # Initialize Alpaca clients and subscribe
            self._init_clients()
            self._subscribe_handlers()

            # Register signal handlers for graceful shutdown
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.add_signal_handler(sig, lambda s=sig: asyncio.ensure_future(self.shutdown(s)))
                except NotImplementedError:
                    pass

            # Close any lingering positions from previous runs
            await self._clean_existing_positions()

            # Weekend prep: warm up ALL data, auto-tune params, train models
            await self._weekend_prep()

            # Launch background tasks
            self._tasks = [
                asyncio.create_task(self._monitor_positions()),
                asyncio.create_task(self._log_account_summary()),
                asyncio.create_task(self._monitor_momentum_picks()),
            ]

            # Seed ML model with recent historical data for instant readiness
            await self._seed_historical_data()

            # Keep running until shutdown (momentum monitor handles all trading via yfinance)
            await asyncio.gather(*self._tasks, return_exceptions=True)

        except asyncio.CancelledError:
            logger.info("Engine run cancelled.")
        except ConfigurationError as exc:
            logger.critical("Configuration error: %s", exc)
        except Exception as exc:
            logger.critical("Unexpected error in main loop: %s", exc)
            logger.debug(traceback.format_exc())
        finally:
            await self.shutdown()


# ============================================================================
# Entry Point
# ============================================================================


def main() -> int:
    """Console entry point for the Multi-Modal Trading Engine.

    Environment variables required:
      ALPACA_API_KEY      Alpaca API key
      ALPACA_SECRET_KEY   Alpaca secret key

    Optional:
      ALPACA_BASE_URL     Defaults to https://paper-api.alpaca.markets
      DRY_RUN             Set to "false" to disable dry-run (default: true)
      PAPER_TRADING       Set to "false" to use live trading (default: true)
      SYMBOLS             Comma-separated list (default: SPY,QQQ,AAPL,MSFT,NVDA)
      ENABLE_LSTM         Set to "true" to enable LSTM path (default: false)
    """
    # Check required env vars
    if not alpaca_config.api_key or not alpaca_config.secret_key:
        print("ERROR: ALPACA_API_KEY and ALPACA_SECRET_KEY must be set.")
        print("Usage: ALPACA_API_KEY=xxx ALPACA_SECRET_KEY=yyy python trading_engine.py")
        return 1

    # Override config from environment (one-shot overrides)
    dry_run = os.getenv("DRY_RUN", "true").lower() == "true"
    paper_trading = os.getenv("PAPER_TRADING", "true").lower() == "true"
    enable_lstm = os.getenv("ENABLE_LSTM", "false").lower() == "true"

    symbols_env = os.getenv("SYMBOLS")
    if symbols_env:
        trading_config.symbols = [s.strip() for s in symbols_env.split(",")]

    trading_config.dry_run = dry_run
    trading_config.paper_trading = paper_trading

    engine = MultiModalTradingEngine(
        alpaca_cfg=alpaca_config,
        trading_cfg=trading_config,
        model_cfg=model_config,
        sentiment_cfg=sentiment_config,
        sys_cfg=system_config,
    )

    if enable_lstm:
        engine.enable_lstm()

    try:
        asyncio.run(engine.run())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    finally:
        logger.info("Engine process terminated.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
