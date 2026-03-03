"""
PA 仪表板 Web 服务
"""
from flask import Flask, render_template, jsonify, request
from core.pa_engine import run_full_analysis
import traceback

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze')
def analyze():
    symbol = request.args.get('symbol', 'AAPL')
    interval = request.args.get('interval', '1d')
    period = request.args.get('period', '6mo')
    try:
        result = run_full_analysis(symbol, interval, period)
        return jsonify({'ok': True, 'data': result})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e), 'trace': traceback.format_exc()})

if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')
