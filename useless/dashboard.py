"""Flask dashboard for crypto simulator — reads /tmp/crypto_status.json."""
import json, os, threading, webbrowser
from flask import Flask, jsonify, render_template_string

STATUS_PATH = "/tmp/crypto_status.json"
app = Flask(__name__)

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Crypto Sim Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace; background: #0d1117; color: #c9d1d9; padding: 20px; }
.container { max-width: 1200px; margin: 0 auto; }
.header { text-align: center; margin-bottom: 24px; }
.header h1 { font-size: 28px; background: linear-gradient(135deg, #58a6ff, #bc8cff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.header p { color: #8b949e; font-size: 14px; margin-top: 4px; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-bottom: 20px; }
.card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
.card label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: #8b949e; }
.card .value { font-size: 24px; font-weight: 700; margin-top: 4px; }
.card .value.green { color: #3fb950; }
.card .value.red { color: #f85149; }
.card .value.white { color: #c9d1d9; }
.card .sub { font-size: 12px; color: #8b949e; margin-top: 2px; }
.position-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin-bottom: 20px; display: none; }
.position-card.active { display: block; }
.position-card .pos-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
.position-card .pos-symbol { font-size: 20px; font-weight: 700; }
.position-card .pos-gain { font-size: 18px; font-weight: 700; }
.position-card .pos-details { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
.position-card .pos-details div { font-size: 13px; }
.position-card .pos-details span { color: #8b949e; }
.chart-container { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin-bottom: 20px; }
.chart-container h3 { font-size: 14px; color: #8b949e; margin-bottom: 12px; }
.trades-table { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
.trades-table h3 { font-size: 14px; color: #8b949e; margin-bottom: 12px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { text-align: left; padding: 8px 12px; border-bottom: 1px solid #30363d; color: #8b949e; font-weight: 500; }
td { padding: 8px 12px; border-bottom: 1px solid #21262d; }
tr:last-child td { border-bottom: none; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; }
.badge-win { background: rgba(63, 185, 80, 0.15); color: #3fb950; }
.badge-loss { background: rgba(248, 81, 73, 0.15); color: #f85149; }
.badge-exit { background: rgba(139, 148, 158, 0.15); color: #8b949e; }
.params { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px 16px; margin-bottom: 20px; display: flex; gap: 16px; flex-wrap: wrap; font-size: 12px; color: #8b949e; }
.params span { color: #c9d1d9; font-weight: 600; }
@media (max-width: 600px) {
  .cards { grid-template-columns: repeat(2, 1fr); }
  .position-card .pos-details { grid-template-columns: repeat(2, 1fr); }
}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>🚀 Crypto Sim Dashboard</h1>
    <p id="subtitle">Loading...</p>
  </div>

  <div class="params" id="params"></div>

  <div class="cards" id="cards">
    <div class="card"><label>Equity</label><div class="value white" id="equity">--</div></div>
    <div class="card"><label>P&L</label><div class="value" id="pnl">--</div><div class="sub" id="pnl_pct"></div></div>
    <div class="card"><label>Peak</label><div class="value white" id="peak">--</div></div>
    <div class="card"><label>Trades</label><div class="value white" id="trades">--</div></div>
  </div>

  <div class="position-card" id="position-card">
    <div class="pos-header">
      <div class="pos-symbol" id="pos-symbol"></div>
      <div class="pos-gain" id="pos-gain"></div>
    </div>
    <div class="pos-details">
      <div><span>Entry:</span> <strong id="pos-entry"></strong></div>
      <div><span>Current:</span> <strong id="pos-current"></strong></div>
      <div><span>SL:</span> <strong id="pos-sl"></strong></div>
      <div><span>Quantity:</span> <strong id="pos-qty"></strong></div>
    </div>
  </div>

  <div class="chart-container">
    <h3>📈 Equity Curve</h3>
    <canvas id="equity-chart" height="120"></canvas>
  </div>

  <div class="trades-table">
    <h3>📋 Recent Trades</h3>
    <table><thead><tr>
      <th>Symbol</th><th>Entry</th><th>Exit</th><th>P&L</th><th>Reason</th><th>Time</th>
    </tr></thead><tbody id="trades-body"></tbody></table>
  </div>
</div>

<script>
let chart = null;
let equityData = [];

function fmt(n) { return (n || 0).toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2}); }
function fmt6(n) { return (n || 0).toLocaleString('en-US', {minimumFractionDigits: 4, maximumFractionDigits: 6}); }

async function refresh() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    document.getElementById('subtitle').textContent = 'Last updated: ' + new Date(d.timestamp).toLocaleTimeString();

    document.getElementById('equity').textContent = '$' + fmt(d.equity);
    document.getElementById('peak').textContent = '$' + fmt(d.peak_equity);

    const pnl = document.getElementById('pnl');
    const pnlPct = document.getElementById('pnl_pct');
    pnl.textContent = (d.total_pnl >= 0 ? '+' : '') + '$' + fmt(d.total_pnl);
    pnl.className = 'value ' + (d.total_pnl >= 0 ? 'green' : 'red');
    pnlPct.textContent = (d.total_pnl_pct >= 0 ? '+' : '') + fmt(d.total_pnl_pct) + '%';

    document.getElementById('trades').textContent = d.total_trades;

    // Params
    if (d.params) {
      document.getElementById('params').innerHTML = Object.entries(d.params).map(([k,v]) =>
        `<div>${k.replace(/_/g,' ')}: <span>${v}</span></div>`
      ).join('');
    }

    // Position
    const pc = document.getElementById('position-card');
    if (d.position) {
      pc.classList.add('active');
      document.getElementById('pos-symbol').textContent = d.position.symbol;
      const g = document.getElementById('pos-gain');
      g.textContent = (d.position.gain_pct >= 0 ? '+' : '') + fmt(d.position.gain_pct) + '%';
      g.className = 'pos-gain ' + (d.position.gain_pct >= 0 ? 'green' : 'red');
      document.getElementById('pos-entry').textContent = '$' + fmt6(d.position.entry_price);
      document.getElementById('pos-current').textContent = '$' + fmt6(d.position.current_price);
      document.getElementById('pos-sl').textContent = '$' + fmt6(d.position.sl_price);
      document.getElementById('pos-qty').textContent = fmt(d.position.quantity);
    } else {
      pc.classList.remove('active');
    }

    // Trades
    const tb = document.getElementById('trades-body');
    if (d.trades && d.trades.length) {
      tb.innerHTML = d.trades.map(t => {
        const cls = t.gain_pct > 0 ? 'badge-win' : t.gain_pct < 0 ? 'badge-loss' : 'badge-exit';
        return `<tr>
          <td><strong>${t.symbol}</strong></td>
          <td>$${fmt(t.entry)}</td>
          <td>$${fmt(t.exit)}</td>
          <td><span class="badge ${cls}">${t.gain_pct >= 0 ? '+' : ''}${fmt(t.gain_pct)}%</span></td>
          <td>${t.reason}</td>
          <td>${new Date(t.time).toLocaleTimeString()}</td>
        </tr>`;
      }).join('');
    } else {
      tb.innerHTML = '<tr><td colspan="6" style="text-align:center;color:#8b949e;padding:20px;">No trades yet</td></tr>';
    }

    // Equity chart
    if (d.equity_history && d.equity_history.length > 1) {
      equityData = d.equity_history;
      const labels = equityData.map(p => new Date(p.t * 1000).toLocaleTimeString());
      const values = equityData.map(p => p.e);
      if (chart) {
        chart.data.labels = labels;
        chart.data.datasets[0].data = values;
        chart.update('none');
      } else {
        const ctx = document.getElementById('equity-chart').getContext('2d');
        chart = new Chart(ctx, {
          type: 'line',
          data: { labels, datasets: [{
            label: 'Equity',
            data: values,
            borderColor: '#58a6ff',
            backgroundColor: 'rgba(88, 166, 255, 0.1)',
            borderWidth: 2,
            fill: true,
            tension: 0.3,
            pointRadius: 0,
          }]},
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
              x: { display: true, ticks: { color: '#8b949e', maxTicksLimit: 8, font: { size: 10 } } },
              y: { display: true, ticks: { color: '#8b949e', font: { size: 10 }, callback: v => '$' + v } },
            },
            animation: false,
          }
        });
      }
    }
  } catch(e) {
    document.getElementById('subtitle').textContent = 'Waiting for crypto sim to start...';
  }
}

setInterval(refresh, 2000);
refresh();
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/api/status")
def api_status():
    try:
        with open(STATUS_PATH) as f:
            return jsonify(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        return jsonify({"error": "No data yet"})

if __name__ == "__main__":
    port = 5050
    print(f"🌐 Dashboard at http://localhost:{port}")
    threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    app.run(host="0.0.0.0", port=port, debug=False)
