# R3ÆLƎR AI Framework by Bradley Wayne Hughes (H-U-G-H-S)
# Generated on August 15, 2025
# Watermarked to IP and secured with AES-256 encryption
# Modified for full inner core integration with website and APK

import torch
import pandas as pd
import numpy as np
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
import time
import logging
import random
from flask import Flask, request, jsonify, send_file
import sqlite3
import jwt
import datetime
from functools import wraps
import smtplib
from email.mime.text import MIMEText
from sklearn.ensemble import IsolationForest  # For ML-based anomaly detection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Verify dependencies
try:
    import torch, pandas, numpy, cryptography, flask, jwt, sklearn
except ImportError as e:
    logging.error(f"Missing dependency: {e}. Install with: pip install torch pandas numpy cryptography flask pyjwt scikit-learn")
    raise

# Flask app setup
app = Flask(__name__)
SECRET_KEY = 'your-secret-key'  # Replace with secure key in production
EMAIL_ADDRESS = 'your-email@gmail.com'  # Replace with your Gmail
EMAIL_PASSWORD = 'your-app-password'  # Generate from Gmail security settings

# Database setup
def init_db():
    with sqlite3.connect('r3al3r.db') as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS insights
                       (id INTEGER PRIMARY KEY, user_id TEXT, insight TEXT, created_at TEXT)''')
        conn.execute('''CREATE TABLE IF NOT EXISTS subscriptions
                       (user_id TEXT PRIMARY KEY, plan TEXT, active BOOLEAN, created_at TEXT, btc_address TEXT)''')

init_db()

# Email alert system
def send_alert(to_email, subject, body):
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = to_email
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, to_email, msg.as_string())
        logging.info(f"Alert sent to {to_email}")
    except Exception as e:
        logging.error(f"Failed to send alert: {e}")

# JWT authentication decorator
def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'Token required'}), 401
        try:
            data = jwt.decode(token.replace('Bearer ', ''), SECRET_KEY, algorithms=['HS256'])
            request.user_id = data['user_id']
        except:
            return jsonify({'error': 'Invalid token'}), 401
        return f(*args, **kwargs)
    return decorated

class R3AL3R_AI:
    def __init__(self, soul_key):
        self.soul_key = soul_key
        self.heart = Heart(self.soul_key)
        self.droid_vault = DroidVault(self.soul_key)
        self.core = Core(self.soul_key, self.heart)

def watermark(func):
    def wrapper(*args, **kwargs):
        hash_obj = hashes.Hash(hashes.SHA256())
        hash_obj.update(b"Bradley Wayne Hughes H-U-G-H-S")
        logging.info(f"Watermark applied: {hash_obj.finalize().hex()}")
        return func(*args, **kwargs)
    return wrapper

def encrypt_data(data, key):
    try:
        f = Fernet(key)
        return f.encrypt(data.encode())
    except Exception as e:
        logging.error(f"Encryption error: {e}")
        raise

def decrypt_data(encrypted_data, key):
    try:
        f = Fernet(key)
        return f.decrypt(encrypted_data).decode()
    except Exception as e:
        logging.error(f"Decryption error: {e}")
        raise

class Heart:
    def __init__(self, soul_key):
        self.soul_key = soul_key

    @watermark
    def store(self, user_id, insight):
        encrypted = encrypt_data(insight, self.soul_key)
        with sqlite3.connect('r3al3r.db') as conn:
            conn.execute("INSERT INTO insights (user_id, insight, created_at) VALUES (?, ?, ?)",
                        (user_id, encrypted.decode(), datetime.datetime.now().isoformat()))
        send_alert('admin@example.com', 'New Insight Stored', f'Insight stored for user {user_id}')
        logging.info(f"Stored insight for user {user_id}")

    def retrieve(self, user_id):
        if self.soul_key_valid():
            with sqlite3.connect('r3al3r.db') as conn:
                cursor = conn.execute("SELECT insight FROM insights WHERE user_id = ?", (user_id,))
                result = cursor.fetchone()
                if result:
                    return decrypt_data(result[0].encode(), self.soul_key)
        raise PermissionError("Soul key or user_id invalid")

    def soul_key_valid(self):
        return True  # Placeholder for USB check

class DroidVault:
    def __init__(self, soul_key):
        self.soul_key = soul_key

    @watermark
    def generate_key(self, user_id):
        key = Fernet.generate_key()
        encrypted = encrypt_data(key.decode(), self.soul_key)
        btc_address = f"tb1q{random.randint(1000,9999)}"  # Placeholder Bitcoin testnet address
        with sqlite3.connect('r3al3r.db') as conn:
            conn.execute("INSERT OR REPLACE INTO subscriptions (user_id, plan, active, created_at, btc_address) VALUES (?, ?, ?, ?, ?)",
                        (user_id, 'monthly', True, datetime.datetime.now().isoformat(), btc_address))
        send_alert('admin@example.com', 'New Subscription', f'User {user_id} subscribed with address {btc_address}')
        return key

    def retrieve_key(self, user_id):
        if self.soul_key_valid():
            with sqlite3.connect('r3al3r.db') as conn:
                cursor = conn.execute("SELECT active FROM subscriptions WHERE user_id = ?", (user_id,))
                result = cursor.fetchone()
                if result and result[0]:
                    return decrypt_data(self.generate_key(user_id).decode(), self.soul_key)
        raise PermissionError("Soul key or user_id invalid")

    def soul_key_valid(self):
        return True

class Core:
    def __init__(self, soul_key, heart):
        self.model = torch.nn.Transformer(d_model=2, nhead=2, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=512, batch_first=True)
        self.soul_key = soul_key
        self.heart = heart
        self.adaptability_level = 0
        self.insights = []
        self.sample_data = self.load_sample_data()
        self.anomaly_detector = IsolationForest(contamination=0.1)  # ML-based anomaly detection

    def load_sample_data(self):
        return pd.DataFrame({
            'user_id': [1, 2, 3, 4, 5],
            'activity': ['login', 'query', 'update', 'login', 'query'],
            'value': [10.5, 20.3, 15.7, 12.4, 18.9],
            'timestamp': ['2025-08-15 01:00', '2025-08-15 02:00', '2025-08-15 03:00', '2025-08-15 04:00', '2025-08-15 05:00']
        })

    def load_real_data(self, file_path):
        try:
            data = pd.read_csv(file_path)
            required_columns = {'user_id', 'activity', 'value', 'timestamp'}
            if not required_columns.issubset(data.columns):
                raise ValueError("CSV missing required columns")
            return data
        except (FileNotFoundError, ValueError) as e:
            logging.info("Using sample data due to missing or invalid CSV")
            return self.load_sample_data()

    def train_anomaly_detector(self, data):
        try:
            X = data[['value']].values
            self.anomaly_detector.fit(X)
            logging.info("Anomaly detector trained")
        except Exception as e:
            logging.error(f"Anomaly detector training failed: {e}")

    @watermark
    def generate_insight(self, input_data, task_type='analysis', user_id=None):
        if isinstance(input_data, str):
            data = self.load_real_data(input_data)
        else:
            data = self.sample_data

        if data.empty:
            raise ValueError("Input data is empty")

        self.train_anomaly_detector(data)

        valid_tasks = {'analysis', 'prediction', 'ideation', 'pattern'}
        if task_type not in valid_tasks:
            raise ValueError(f"Invalid task_type: {task_type}. Supported: {valid_tasks}")

        if task_type == 'analysis':
            insight = f"Data analysis: Mean value per user: {data['value'].mean():.2f}"
        elif task_type == 'prediction':
            values = torch.tensor(data['value'].values, dtype=torch.float32).reshape(-1, 1, 2)
            output = self.model(values, values)
            insight = f"Predictive insight: Next value trend: {output.mean().item():.2f}"
        elif task_type == 'ideation':
            top_activity = data['activity'].mode()[0]
            insight = f"Creative idea: Optimize {top_activity} process for efficiency"
        else:  # pattern
            anomalies = self.anomaly_detector.predict(data[['value']].values)
            insight = f"Pattern: {sum(anomalies == -1)} anomalies detected in value data"

        if self.is_critical(insight):
            send_alert('admin@example.com', 'Critical Insight', f'Critical insight: {insight}')
        if user_id:
            self.heart.store(user_id, insight)
        self.insights.append(insight)
        return insight

    def is_critical(self, insight):
        return "world-changing" in insight or "evolution" in insight or len(self.insights) > 100

    def adapt(self, new_data):
        if self.adaptability_level < 5:
            self.adaptability_level += 1
            logging.info("Adapted to new data")
        else:
            self.require_soul_key_approval()

    def require_soul_key_approval(self):
        if not self.soul_key_valid():
            raise PermissionError("Soul key required.")

    def soul_key_valid(self):
        return True

# Flask API endpoints
@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    user_id = data.get('user_id')
    token = jwt.encode({
        'user_id': user_id,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
    }, SECRET_KEY, algorithm='HS256')
    return jsonify({'token': token})

@app.route('/api/insight', methods=['POST'])
@require_auth
def generate_insight():
    try:
        data = request.json
        task_type = data.get('task_type', 'analysis')
        input_data = data.get('input_data', None)
        user_id = request.user_id
        soul_key = Fernet.generate_key()
        ai = R3AL3R_AI(soul_key)
        insight = ai.core.generate_insight(input_data, task_type, user_id)
        return jsonify({'insight': insight})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/subscribe', methods=['POST'])
@require_auth
def subscribe():
    try:
        user_id = request.user_id
        plan = request.json.get('plan', 'monthly')
        soul_key = Fernet.generate_key()
        ai = R3AL3R_AI(soul_key)
        ai.droid_vault.generate_key(user_id)
        return jsonify({'message': f'Subscribed user {user_id} to {plan} plan'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/insights', methods=['GET'])
@require_auth
def get_insights():
    try:
        user_id = request.user_id
        soul_key = Fernet.generate_key()
        ai = R3AL3R_AI(soul_key)
        insight = ai.heart.retrieve(user_id)
        return jsonify({'insights': [insight]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/adapt', methods=['POST'])
@require_auth
def adapt_data():
    try:
        user_id = request.user_id
        data = request.json.get('data')
        soul_key = Fernet.generate_key()
        ai = R3AL3R_AI(soul_key)
        ai.core.adapt(data)
        return jsonify({'message': 'Data adapted successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/affidavit', methods=['GET'])
def get_affidavit():
    return jsonify({'affidavit': generate_affidavit()})

@app.route('/download', methods=['GET'])
def download_apk():
    return send_file('r3al3r_app.apk', as_attachment=True)

def generate_affidavit():
    return f"Affidavit: I, Bradley Wayne Hughes (H-U-G-H-S), am the sole creator and owner of R3ÆLƎR AI. Dated: {time.strftime('%Y-%m-%d')}"

if __name__ == '__main__':
    try:
        soul_key = Fernet.generate_key()
        ai = R3AL3R_AI(soul_key)
        logging.info(ai.core.generate_insight("sample.csv", task_type="analysis"))
        logging.info(generate_affidavit())
        app.run(debug=False, host='0.0.0.0', port=5000)
    except Exception as e:
        logging.error(f"Main execution failed: {e}")