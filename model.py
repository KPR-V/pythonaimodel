import os
import json
import joblib
import pandas as pd
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import requests
import json

app = Flask(__name__)
CORS(app)

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

# Feature columns expected by the model
FEATURES = [
    'avg_loss_making_trades',
    'avg_profitable_trades',
    'collection_score',
    'diamond_hands',
    'fear_and_greed_index',
    'holder_metrics_score',
    'liquidity_score',
    'loss_making_trades',
    'loss_making_trades_percentage',
    'loss_making_volume',
    'market_dominance_score',
    'metadata_score',
    'profitable_trades',
    'profitable_trades_percentage',
    'profitable_volume',
    'token_distribution_score',
    'washtrade_index'
]

# Paths to your serialized artifacts
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'isolation_forest_model.joblib')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.joblib')

# Load model & scaler
clf = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Load your NFT-data API key from env
UNLEASHNFTS_API_KEY = os.getenv('UNLEASHNFTS_API_KEY')
if not UNLEASHNFTS_API_KEY:
    raise RuntimeError("Missing UNLEASHNFTS_API_KEY environment variable")

# ------------------------------------------------------------------------------
# Core Logic
# ------------------------------------------------------------------------------

def calculate_risk_score(df: pd.DataFrame) -> tuple[float, str]:
    """Normalize IsolationForest score into [0–100] and bucket into risk categories."""
    raw = clf.decision_function(scaler.transform(df))
    # Adjust these min/max bounds to your training distribution
    norm = 100 * (1 - (raw - (-0.26)) / (0.16 - (-0.26)))
    score = float(norm[0])

    if score < 10:
        cat = "Low Risk"
    elif score < 60:
        cat = "Medium Risk"
    else:
        cat = "High Risk"
    return score, cat

def identify_risk_factors(df: pd.DataFrame) -> list[str]:
    """Flag individual risk factors based on static thresholds."""
    row = df.iloc[0]
    factors = []
    if row.washtrade_index > 50:
        factors.append("High wash trading activity")
    if row.loss_making_trades_percentage > 70:
        factors.append("High % of loss-making trades")
    if row.liquidity_score < 30:
        factors.append("Low liquidity")
    if row.holder_metrics_score < 40:
        factors.append("Poor holder metrics")
    return factors

def predict_risk_for_contract(address: str) -> dict:
    """Fetch on-chain metrics, run the model, and return risk output."""
    url = (
        "https://api.unleashnfts.com/api/v2/nft/collection/profile"
        f"?blockchain=ethereum&contract_address={address}"
        "&offset=0&limit=1&sort_by=washtrade_index&time_range=all&sort_order=desc"
    )
    headers = {
        "Accept": "application/json",
        "x-api-key": UNLEASHNFTS_API_KEY
    }
    resp = requests.get(url, headers=headers, timeout=10)
    data = resp.json().get("data")
    if not data:
        return {"error": "No data available for this contract"}

    df = pd.DataFrame([data[0]], columns=FEATURES)
    score, category = calculate_risk_score(df)
    return {
        "risk_score": round(score, 2),
        "risk_category": category,
        "contributing_factors": identify_risk_factors(df)
    }

# ------------------------------------------------------------------------------
# HTTP Endpoints
# ------------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200

@app.route("/api", methods=["GET"])
def index():
    return jsonify({"message": "NFT Risk Predictor is running"}), 200

@app.route("/api/predict-risk", methods=["POST"])
def predict_risk_route():
    body = request.get_json(force=True)
    address = body.get("contract_address")
    if not address:
        return jsonify({"error": "Please provide a contract_address"}), 400

    try:
        result = predict_risk_for_contract(address)
        return jsonify(result), (200 if "error" not in result else 502)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # Local dev: uses PORT env var or defaults to 5012
    port = int(os.getenv("PORT", 5012))
    app.run(host="0.0.0.0", port=port, debug=False)
