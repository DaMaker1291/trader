"""
Hugging Face Fade Prediction Model — optional AI backend for gap_bot_v5.

Purpose:
  A small RandomForest model trained on live trade data to predict
  whether a gap-up will "fade" (reverse) or "run" (continue higher).

Architecture:
  - Trained on features: gap%, RVOL trend, first 3 bar direction, volume
  - Inference <1ms via sklearn (no HF inference API needed for speed)
  - Can be exported to ONNX or hosted as HF Space for API access

Files:
  hf_fade_model.py — train/predict/export
  hf_space/        — optional HF Space deployment

Usage:
  python3 hf_fade_model.py --train       # train on /tmp/gap_trades_v5.jsonl
  python3 hf_fade_model.py --predict ... # predict single sample
  python3 hf_fade_model.py --export      # export to ONNX / HF Space
  
HF Space Deployment:
  1. python3 hf_fade_model.py --export
  2. Upload model.pkl + app.py to https://huggingface.co/new-space
  3. Set HF_MODEL_URL=https://your-space.hf.space/predict in env
"""

import json, os, sys, pickle, random
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

FEATURE_NAMES = [
    "gap_pct", "rvol_trend", "first_bar_dir", "second_bar_dir",
    "third_bar_dir", "pre_vol", "rel_vol", "price", "weekday",
]


def extract_features(trade: Dict) -> List[float]:
    """Extract feature vector from a trade record."""
    gap = trade.get("gap_pct", trade.get("gap", 0))
    return [
        gap,
        trade.get("rvol_trend", 0),
        1 if trade.get("first_bar_dir", 0) > 0 else -1 if trade.get("first_bar_dir", 0) < 0 else 0,
        1 if trade.get("second_bar_dir", 0) > 0 else -1 if trade.get("second_bar_dir", 0) < 0 else 0,
        1 if trade.get("third_bar_dir", 0) > 0 else -1 if trade.get("third_bar_dir", 0) < 0 else 0,
        trade.get("pre_vol", trade.get("vol", 50000)),
        trade.get("rel_vol", 1.0),
        trade.get("price", 50),
        trade.get("weekday", datetime.now().weekday()),
    ]


def load_trades(path: str = "/tmp/gap_trades_v5.jsonl") -> List[Dict]:
    """Load trades from JSONL file."""
    trades = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        trades.append(json.loads(line))
                    except Exception:
                        pass
    return trades


def prepare_dataset(trades: List[Dict]) -> Tuple[List, List]:
    """Convert trade list to X, y arrays."""
    X, y = [], []
    for t in trades:
        # Label: win (gain > 0) or loss
        gain = t.get("gain_pct", t.get("gain", 0))
        if gain == 0:
            continue
        X.append(extract_features(t))
        y.append(1 if gain > 0 else 0)
    return X, y


def train(trades_path: str = "/tmp/gap_trades_v5.jsonl",
          model_path: str = "/tmp/fade_model.pkl"):
    """Train a RandomForest classifier on trade data.
    
    Falls back to simpler model if sklearn not available.
    """
    trades = load_trades(trades_path)
    if len(trades) < 50:
        # Seed with simulated data for cold-start
        trades = _seed_trades(200)
        print(f"Seeded {len(trades)} trades for training")

    X, y = prepare_dataset(trades)
    print(f"Training on {len(X)} samples ({sum(y)} wins, {len(y) - sum(y)} losses)")

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        model = RandomForestClassifier(
            n_estimators=100, max_depth=6,
            min_samples_leaf=5, random_state=42,
            class_weight="balanced",
        )
        model.fit(X, y)

        # Feature importance
        importances = sorted(zip(FEATURE_NAMES, model.feature_importances_),
                           key=lambda x: -x[1])
        print("\nFeature importance:")
        for name, imp in importances[:6]:
            print(f"  {name:20s}: {imp:.3f}")

        # Save
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"\nModel saved to {model_path}")

        # Also save feature names for inference
        meta_path = model_path.replace(".pkl", "_meta.json")
        with open(meta_path, "w") as f:
            json.dump({"features": FEATURE_NAMES}, f)

        return model

    except ImportError:
        print("sklearn not installed. Using simple logistic model.")
        return _train_simple(X, y, model_path)


def _train_simple(X: List, y: List, model_path: str):
    """Fallback: count-based probability estimator."""
    # Group by binned gap% and first bar direction
    stats = defaultdict(lambda: {"w": 0, "l": 0})
    for features, label in zip(X, y):
        gap_bin = str(int(features[0] / 5) * 5)
        first_dir = features[2]
        key = f"gap={gap_bin}_dir={first_dir}"
        if label:
            stats[key]["w"] += 1
        else:
            stats[key]["l"] += 1

    model = {"type": "simple", "stats": dict(stats), "total_w": sum(y), "total": len(y)}
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Simple model saved to {model_path}")
    return model


def _seed_trades(n: int = 200) -> List[Dict]:
    """Generate synthetic trades for cold-start training."""
    random.seed(42)
    trades = []
    for _ in range(n):
        gap = random.uniform(5, 25)
        vol = random.randint(50000, 5000000)
        rel_vol = random.uniform(0.5, 5.0)
        rvol_trend = random.uniform(-0.3, 0.3)
        first_dir = random.choice([-1, 0, 1])
        second_dir = random.choice([-1, 0, 1])
        third_dir = random.choice([-1, 0, 1])
        price = random.uniform(10, 150)
        weekday = random.randint(0, 4)

        # Label: synthetic win probability
        fade_indicators = (first_dir < 0) + (second_dir < 0) + (third_dir < 0)
        win_prob = 0.35 + (gap / 50) * 0.25 - fade_indicators * 0.10
        win_prob = max(0.1, min(win_prob, 0.85))
        is_win = random.random() < win_prob

        gain = random.uniform(3, 20) if is_win else -random.uniform(1, 5)
        trades.append({
            "gap_pct": gap, "gap": gap, "gain_pct": gain, "gain": gain,
            "rvol_trend": rvol_trend, "first_bar_dir": first_dir,
            "second_bar_dir": second_dir, "third_bar_dir": third_dir,
            "pre_vol": vol, "vol": vol, "rel_vol": rel_vol,
            "price": price, "weekday": weekday,
            "simulated": True,
        })
    return trades


def predict(features: List[float], model_path: str = "/tmp/fade_model.pkl") -> float:
    """Predict fade probability for a single sample."""
    if not os.path.exists(model_path):
        return 0.5

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    if isinstance(model, dict) and model.get("type") == "simple":
        # Simple model: look up by gap bin + first bar direction
        gap_bin = str(int(features[0] / 5) * 5)
        first_dir = features[2]
        key = f"gap={gap_bin}_dir={first_dir}"
        stats = model["stats"].get(key, {"w": 0, "l": 0})
        total = stats["w"] + stats["l"]
        if total >= 5:
            wr = stats["w"] / total
        else:
            wr = model["total_w"] / max(model["total"], 1)
        # Convert win rate to fade probability
        return round(1 - wr, 3)

    # sklearn model
    try:
        proba = model.predict_proba([features])
        # proba[0][1] = probability of win
        # fade_prob = 1 - win_prob
        fade_prob = 1 - proba[0][1]
        return round(float(fade_prob), 3)
    except Exception:
        return 0.5


def export_hf_space(model_path: str = "/tmp/fade_model.pkl",
                    space_dir: str = "hf_space"):
    """Export model + inference code for Hugging Face Space deployment."""
    import shutil

    os.makedirs(space_dir, exist_ok=True)

    # Copy model
    if os.path.exists(model_path):
        shutil.copy(model_path, os.path.join(space_dir, "model.pkl"))

    # Write app.py
    app_code = '''"""
HF Space: Fade Prediction API for gap_bot_v5.

Deploy at https://huggingface.co/new-space
Uses Gradio for UI + FastAPI-compatible /predict endpoint.

Usage:
  import requests
  r = requests.post("https://your-space.hf.space/predict", json={...})
  print(r.json())  # {"fade_prob": 0.32}
"""
import json, pickle, os
import numpy as np
import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

FEATURES = ["gap_pct", "rvol_trend", "first_bar_dir", "second_bar_dir",
            "third_bar_dir", "pre_vol", "rel_vol", "price", "weekday"]


class PredictRequest(BaseModel):
    symbol: str = ""
    gap: float = 5.0
    price: float = 50.0
    rvol_trend: float = 0.0
    first_bar: float = 0.0
    second_bar: float = 0.0
    third_bar: float = 0.0
    pre_vol: float = 50000.0
    rel_vol: float = 1.0
    weekday: int = 2


@app.post("/predict")
async def predict(req: PredictRequest):
    if model is None:
        return {"fade_prob": 0.5, "error": "no model"}

    # Build feature vector
    first_dir = 1 if req.first_bar > 0 else -1 if req.first_bar < 0 else 0
    second_dir = 1 if req.second_bar > 0 else -1 if req.second_bar < 0 else 0
    third_dir = 1 if req.third_bar > 0 else -1 if req.third_bar < 0 else 0

    features = [
        req.gap, req.rvol_trend, first_dir, second_dir, third_dir,
        req.pre_vol, req.rel_vol, req.price, req.weekday,
    ]

    try:
        if isinstance(model, dict) and model.get("type") == "simple":
            gap_bin = str(int(features[0] / 5) * 5)
            key = f"gap={gap_bin}_dir={features[2]}"
            stats = model["stats"].get(key, {"w": 0, "l": 0})
            total = stats["w"] + stats["l"]
            wr = stats["w"] / total if total >= 5 else model["total_w"] / max(model["total"], 1)
            fade_prob = round(1 - wr, 3)
        else:
            proba = model.predict_proba([features])
            fade_prob = round(float(1 - proba[0][1]), 3)
    except Exception as e:
        return {"fade_prob": 0.5, "error": str(e)}

    return {"fade_prob": fade_prob, "symbol": req.symbol}


# Gradio UI
def predict_ui(gap, rvol_trend, first_bar, second_bar, third_bar,
               pre_vol, rel_vol, price):
    req = PredictRequest(
        gap=gap, rvol_trend=rvol_trend,
        first_bar=first_bar, second_bar=second_bar, third_bar=third_bar,
        pre_vol=pre_vol, rel_vol=rel_vol, price=price,
    )
    import asyncio
    result = asyncio.run(predict(req))
    fade_pct = result.get("fade_prob", 0.5) * 100
    verdict = "FADE" if fade_pct > 50 else "RUN"
    return f"{verdict} (fade probability: {fade_pct:.0f}%)"


iface = gr.Interface(
    fn=predict_ui,
    inputs=[
        gr.Slider(0, 30, value=8, label="Gap %"),
        gr.Slider(-0.5, 0.5, value=0, label="RVOL Trend"),
        gr.Slider(-5, 5, value=0, label="1st Bar %"),
        gr.Slider(-5, 5, value=0, label="2nd Bar %"),
        gr.Slider(-5, 5, value=0, label="3rd Bar %"),
        gr.Number(value=500000, label="Pre-market Volume"),
        gr.Number(value=2.0, label="Relative Volume"),
        gr.Number(value=50, label="Underlying Price"),
    ],
    outputs="text",
    title="Gap Fade Predictor",
    description="Predicts whether a gap-up will fade or run based on early price action.",
)

app = gr.mount_gradio_app(app, iface, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
'''
    with open(os.path.join(space_dir, "app.py"), "w") as f:
        f.write(app_code)

    # Write requirements
    with open(os.path.join(space_dir, "requirements.txt"), "w") as f:
        f.write("fastapi\nuvicorn\ngradio\nscikit-learn\nnumpy\npydantic\n")

    # Write README
    with open(os.path.join(space_dir, "README.md"), "w") as f:
        f.write("""---
title: Gap Fade Predictor
emoji: 
colorFrom: red
colorTo: green
sdk: docker
---
# Gap Fade Predictor

Predicts whether a pre-market gap-up will fade or run.
Used by gap_bot_v5.py for entry filtering.

## API

POST /predict with JSON body:
```json
{
  "gap": 8.5,
  "rvol_trend": 0.1,
  "first_bar": 0.5,
  "second_bar": 1.2,
  "third_bar": 0.8,
  "pre_vol": 500000,
  "rel_vol": 2.5,
  "price": 150.0
}
```

Returns:
```json
{"fade_prob": 0.32, "symbol": ""}
```
""")

    # Write Dockerfile
    with open(os.path.join(space_dir, "Dockerfile"), "w") as f:
        f.write("""FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
""")

    print(f"HF Space exported to {space_dir}/")
    print(f"To deploy: cd {space_dir} && git init && git add . && git commit -m 'init'")
    print("Then push to https://huggingface.co/new-space")


if __name__ == "__main__":
    if "--train" in sys.argv:
        train()
    elif "--predict" in sys.argv:
        # Example prediction
        feats = [8.5, 0.1, 1, 1, 1, 500000, 2.5, 150, 2]
        prob = predict(feats)
        print(f"Fade probability: {prob:.1%}")
    elif "--export" in sys.argv:
        export_hf_space()
    else:
        print(__doc__)
