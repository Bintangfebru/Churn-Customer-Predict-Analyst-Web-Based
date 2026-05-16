# app.py — ChurnPredict Dashboard Backend (OPTIMIZED)
# =========================================================
# PERUBAHAN UTAMA vs versi lama:
#   1. Ganti df.iterrows() → operasi vektorisasi pandas (10-50x lebih cepat)
#   2. Ganti predict_churn() per-baris → batch predict_proba() sekaligus
#   3. Upload async: langsung return, frontend polling /api/upload/status
#   4. Hilangkan definisi fungsi di dalam loop
# =========================================================

from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import os
import threading
import uuid
from werkzeug.utils import secure_filename
import joblib
from datetime import datetime
import json
import traceback

app = Flask(__name__)

# ── Custom JSON encoder: NaN/Inf → null ─────────────────
import math, json as _json
class _SafeJSONProvider(app.json_provider_class):
    def dumps(self, obj, **kw):
        # Serialisasi dulu ke string, lalu ganti NaN/Infinity ke null
        raw = super().dumps(obj, **kw)
        # Ganti literal NaN dan Infinity yang lolos dari encoder standar
        raw = raw.replace(': NaN',  ': null') \
                 .replace(':NaN',   ':null') \
                 .replace(': Infinity',  ': null') \
                 .replace(':Infinity',   ':null') \
                 .replace(': -Infinity', ': null') \
                 .replace(':-Infinity',  ':null')
        return raw

app.json_provider_class = _SafeJSONProvider
app.json = _SafeJSONProvider(app)

# ── Config ──────────────────────────────────────────────
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Global State ────────────────────────────────────────
customers_data = []
current_csv_filename = None
model = None
model_type_name = "Not loaded"

# Status tracker untuk async upload
upload_jobs = {}  # job_id -> { status, progress, message, error }

# ── Feature Mapping ──────────────────────────────────────
FEATURE_MAPPING = {
    "csat_score":           ["csat_score", "csat", "satisfaction", "rating", "score"],
    "payment_failures":     ["payment_failures", "failed_payments", "payment_errors"],
    "tenure_months":        ["tenure_months", "tenure", "months", "subscription_months"],
    "monthly_logins":       ["monthly_logins", "logins_per_month", "login_count", "logins"],
    "total_revenue":        ["total_revenue", "revenue", "lifetime_value", "ltv"],
    "discount_applied":     ["discount_applied", "discount", "has_discount"],
    "survey_response":      ["survey_response", "survey", "feedback"],
    "complaint_type":       ["complaint_type", "issue_type", "complaint_category"],
    "signup_channel":       ["signup_channel", "channel", "acquisition_channel"],
    "contract_type":        ["contract_type", "contract", "billing_cycle"],
    "customer_segment":     ["customer_segment", "segment", "plan", "package", "tier"],
    "nps_score":            ["nps_score", "nps", "net_promoter"],
    "escalations":          ["escalations", "escalated", "priority_count"],
    "avg_resolution_time":  ["avg_resolution_time", "resolution_time", "response_time"],
}

EXTRA_COLUMN_MAPPING = {
    'customer_id':      ['customer_id', 'id', 'customerid', 'cust_id'],
    'name':             ['name', 'customer_name', 'full_name'],
    'gender':           ['gender', 'sex'],
    'age':              ['age', 'customer_age'],
    'country':          ['country', 'negara', 'region'],
    'city':             ['city', 'kota'],
    'monthly_fee':      ['monthly_fee', 'fee', 'price', 'subscription_fee', 'amount', 'monthly_charges'],
    'last_login':       ['last_login_days_ago', 'last_login', 'days_since_login', 'inactive_days'],
    'support_tickets':  ['support_tickets', 'tickets', 'complaints', 'ticket_count'],
    'payment_method':   ['payment_method', 'payment_type', 'payment'],
    'churn':            ['churn', 'churned', 'is_churn', 'churn_status', 'churn_label'],
}

# ── Load Model ───────────────────────────────────────────
MODEL_PATH = "model_churn.pkl"

def load_model():
    global model, model_type_name
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            model_type_name = type(model).__name__
            print(f"✅ Model loaded: {model_type_name} from {MODEL_PATH}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            model = None
    else:
        print(f"⚠️  model_churn.pkl not found — will use rule-based scoring")
        model = None

load_model()

# ── Helpers ──────────────────────────────────────────────
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def find_column(df_columns, candidates):
    lower_cols = {c.lower(): c for c in df_columns}
    for cand in candidates:
        if cand.lower() in lower_cols:
            return lower_cols[cand.lower()]
    return None

# ── Vektorisasi Helpers ──────────────────────────────────
def vec_normalize_segment(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.lower()
    result = pd.Series('Standard', index=series.index)
    result[s.str.contains('premium|gold|platinum|vip', na=False)] = 'Premium'
    result[s.str.contains('basic|bronze|starter|free', na=False)] = 'Basic'
    return result

def vec_normalize_contract(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.lower()
    result = pd.Series('Monthly', index=series.index)
    result[s.str.contains('annual|yearly|year|tahunan', na=False)] = 'Annual'
    result[s.str.contains('quarter|tri|3 month|3month|kuartal', na=False)] = 'Quarterly'
    return result

def vec_normalize_bool(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    return s.isin(['1', 'yes', 'true', 'y', 'ya']).astype(int)

def vec_safe_float(series: pd.Series, default=0.0) -> pd.Series:
    return pd.to_numeric(series, errors='coerce').fillna(default)

def vec_safe_int(series: pd.Series, default=0) -> pd.Series:
    return pd.to_numeric(series, errors='coerce').fillna(default).astype(int)

# ── Rule-based Churn (Vektorisasi) ───────────────────────
def rule_based_churn_prob_vectorized(df: pd.DataFrame) -> pd.Series:
    """Hitung churn probability untuk seluruh DataFrame sekaligus (tanpa loop)."""
    score = pd.Series(0.0, index=df.index)

    # Segment
    score += np.where(df['customer_segment'] == 'Basic', 0.14,
             np.where(df['customer_segment'] == 'Standard', 0.07, 0.02))

    # Contract
    score += np.where(df['contract_type'] == 'Monthly', 0.08,
             np.where(df['contract_type'] == 'Annual', -0.10, 0.0))

    # Support tickets
    score += np.where(df['support_tickets'] > 4, 0.20,
             np.where(df['support_tickets'] > 2, 0.08, 0.0))

    # CSAT
    score += np.where(df['csat_score'] < 2.5, 0.22,
             np.where(df['csat_score'] < 3.5, 0.08, 0.0))

    # Logins
    score += np.where(df['monthly_logins'] < 5, 0.15, 0.0)

    # Last login
    score += np.where(df['last_login'] > 30, 0.24,
             np.where(df['last_login'] > 14, 0.10, 0.0))

    # Payment failures
    score += np.where(df['payment_failures'] > 2, 0.15,
             np.where(df['payment_failures'] > 0, 0.05, 0.0))

    # Escalations
    score += np.where(df['escalations'] > 0, 0.08, 0.0)

    # NPS
    score += np.where(df['nps_score'] < 0, 0.06, 0.0)

    # Discount
    score += np.where(df['discount_applied'] == 1, -0.03, 0.0)

    return score.clip(0.02, 0.97)

# ── Batch Prediction ─────────────────────────────────────
def predict_batch(df: pd.DataFrame) -> tuple:
    """
    Prediksi churn untuk seluruh DataFrame sekaligus.
    Return: (probabilities Series, risk Series)
    """
    global model

    if model is not None:
        try:
            seg_map      = {'Premium': 2, 'Standard': 1, 'Basic': 0}
            contract_map = {'Annual': 0, 'Quarterly': 1, 'Monthly': 2}
            channel_keys = ['Organic', 'Referral', 'Paid Ads', 'Social Media']

            feat = pd.DataFrame(index=df.index)
            feat['csat_score']          = vec_safe_float(df.get('csat_score', pd.Series(3.5, index=df.index)), 3.5)
            feat['payment_failures']    = vec_safe_float(df.get('payment_failures', pd.Series(0, index=df.index)), 0)
            feat['tenure_months']       = vec_safe_float(df.get('tenure_months', pd.Series(12, index=df.index)), 12)
            feat['monthly_logins']      = vec_safe_float(df.get('monthly_logins', pd.Series(10, index=df.index)), 10)
            feat['total_revenue']       = vec_safe_float(df.get('total_revenue', pd.Series(0, index=df.index)), 0)
            feat['discount_applied']    = vec_safe_float(df.get('discount_applied', pd.Series(0, index=df.index)), 0)
            feat['survey_response']     = vec_normalize_bool(df.get('survey_response', pd.Series(0, index=df.index))).astype(float)
            feat['complaint_type']      = df.get('complaint_type', pd.Series('None', index=df.index)).astype(str).apply(
                                            lambda x: 0.0 if x in ['None', 'nan', ''] else 1.0)
            feat['signup_channel']      = df.get('signup_channel', pd.Series('Organic', index=df.index)).apply(
                                            lambda x: float(channel_keys.index(x)) if x in channel_keys else 0.0)
            feat['contract_type']       = df.get('contract_type', pd.Series('Monthly', index=df.index)).map(
                                            contract_map).fillna(2).astype(float)
            feat['customer_segment']    = df.get('customer_segment', pd.Series('Standard', index=df.index)).map(
                                            seg_map).fillna(1).astype(float)
            feat['nps_score']           = vec_safe_float(df.get('nps_score', pd.Series(0, index=df.index)), 0)
            feat['escalations']         = vec_safe_float(df.get('escalations', pd.Series(0, index=df.index)), 0)
            feat['avg_resolution_time'] = vec_safe_float(df.get('avg_resolution_time', pd.Series(0, index=df.index)), 0)

            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(feat)[:, 1]
            else:
                probas = model.predict(feat).astype(float)

            probas = np.clip(probas, 0.0, 1.0)
        except Exception as e:
            print(f"  Batch model predict error: {e} — falling back to rule-based")
            probas = rule_based_churn_prob_vectorized(df).values
    else:
        probas = rule_based_churn_prob_vectorized(df).values

    proba_series = pd.Series(probas, index=df.index).round(3)
    risk_series  = pd.cut(proba_series,
                          bins=[-np.inf, 0.35, 0.65, np.inf],
                          labels=['low', 'medium', 'high'])
    return proba_series, risk_series

# ── CSV Processing (VEKTORISASI) ─────────────────────────
def process_customer_data(df: pd.DataFrame, job_id: str = None) -> list:
    """
    Proses seluruh DataFrame dengan operasi vektorisasi pandas.
    TIDAK ADA iterrows() — jauh lebih cepat.
    """
    global customers_data

    def update_job(pct, msg):
        if job_id and job_id in upload_jobs:
            upload_jobs[job_id]['progress'] = pct
            upload_jobs[job_id]['message']  = msg

    print(f"\n📊 CSV columns: {df.columns.tolist()}")
    print(f"📈 Rows: {len(df)}")

    update_job(5, "Mendeteksi kolom...")

    # ── Resolve column names ──────────────────────────────
    all_candidates = {**FEATURE_MAPPING, **EXTRA_COLUMN_MAPPING}
    col_map = {}
    for key, candidates in all_candidates.items():
        found = find_column(df.columns, candidates)
        if found:
            col_map[key] = found

    print(f"✅ Resolved columns: {col_map}")

    # ── Helper: ambil kolom atau buat Series default ──────
    def col(key, default=None):
        if key in col_map:
            return df[col_map[key]]
        if default is None:
            return pd.Series('', index=df.index)
        if callable(default):
            return default()
        return pd.Series(default, index=df.index)

    update_job(15, "Memproses kolom identitas...")

    # ── Identity ──────────────────────────────────────────
    n = len(df)
    auto_ids = pd.Series([f'CUS-{i+1:04d}' for i in range(n)], index=df.index)
    cust_id  = col('customer_id').astype(str).replace(['nan', '', 'None'], np.nan).fillna(auto_ids)

    auto_names = pd.Series([f'Customer {i+1}' for i in range(n)], index=df.index)
    name       = col('name').astype(str).replace(['nan', '', 'None'], np.nan).fillna(auto_names)

    gender  = col('gender', 'Unknown').astype(str).replace(['nan', '', 'None'], 'Unknown')
    country = col('country', 'Unknown').astype(str).replace(['nan', '', 'None'], 'Unknown')
    city    = col('city', 'Unknown').astype(str).replace(['nan', '', 'None'], 'Unknown')
    age     = vec_safe_int(col('age', 30), 30)
    payment = col('payment_method', 'Unknown').astype(str).replace(['nan', '', 'None'], 'Unknown')

    update_job(25, "Memproses segmen & kontrak...")

    # ── Segment / Contract ────────────────────────────────
    segment  = vec_normalize_segment(col('customer_segment', 'Standard'))
    contract = vec_normalize_contract(col('contract_type', 'Monthly'))

    update_job(35, "Memproses data numerik...")

    # ── Numeric ───────────────────────────────────────────
    tenure     = vec_safe_int(col('tenure_months', 12), 12)
    logins     = vec_safe_int(col('monthly_logins', 10), 10)
    last_login = vec_safe_int(col('last_login', 30), 30)
    tickets    = vec_safe_int(col('support_tickets', 0), 0)
    failures   = vec_safe_int(col('payment_failures', 0), 0)
    csat       = vec_safe_float(col('csat_score', 3.5), 3.5).clip(1.0, 5.0)
    escl       = vec_safe_int(col('escalations', 0), 0)
    nps        = vec_safe_int(col('nps_score', 0), 0)
    res_time   = vec_safe_float(col('avg_resolution_time', 0), 0.0)
    discount   = vec_normalize_bool(col('discount_applied', 0))
    survey     = vec_normalize_bool(col('survey_response', 0))

    # Default fee berdasarkan segment
    default_fee = segment.map({'Premium': 250.0, 'Standard': 100.0, 'Basic': 60.0})
    fee        = vec_safe_float(col('monthly_fee'), default=np.nan)
    fee        = fee.where(fee.notna(), default_fee)

    # Default total revenue
    default_rev = fee * tenure
    total_rev   = vec_safe_float(col('total_revenue'), default=np.nan)
    total_rev   = total_rev.where(total_rev.notna(), default_rev)

    update_job(45, "Memproses data kategorik...")

    # ── Categorical ───────────────────────────────────────
    complaint_type = col('complaint_type', 'None').astype(str)
    complaint_type = complaint_type.where(~complaint_type.isin(['nan', '', 'None', 'NaN']), 'None')

    signup_channel = col('signup_channel', 'Organic').astype(str)
    signup_channel = signup_channel.where(~signup_channel.isin(['nan', '', 'None', 'NaN']), 'Organic')

    # ── Churn label ───────────────────────────────────────
    if 'churn' in col_map:
        churn_raw = df[col_map['churn']].astype(str).str.lower()
        churn_label = churn_raw.isin(['1', 'true', 'yes', 'churned', 'churn', 'ya']).astype(int)
        churn_known = ~churn_raw.isin(['nan', '', 'none'])
    else:
        churn_label = pd.Series(0, index=df.index)
        churn_known = pd.Series(False, index=df.index)

    update_job(55, "Menjalankan prediksi AI (batch)...")

    # ── Build working DataFrame untuk prediksi ────────────
    work_df = pd.DataFrame({
        'customer_segment':    segment,
        'contract_type':       contract,
        'support_tickets':     tickets,
        'csat_score':          csat,
        'monthly_logins':      logins,
        'last_login':          last_login,
        'payment_failures':    failures,
        'escalations':         escl,
        'nps_score':           nps,
        'discount_applied':    discount,
        'tenure_months':       tenure,
        'total_revenue':       total_rev,
        'survey_response':     survey,
        'complaint_type':      complaint_type,
        'signup_channel':      signup_channel,
        'avg_resolution_time': res_time,
    }, index=df.index)

    # ── BATCH PREDICTION (satu kali untuk semua baris) ────
    churn_prob, risk = predict_batch(work_df)

    update_job(80, "Menyusun hasil akhir...")

    # Jika churn tidak diketahui, gunakan prediksi
    final_churn = churn_label.copy()
    unknown_mask = ~churn_known
    final_churn[unknown_mask] = (churn_prob[unknown_mask] > 0.5).astype(int)

    # ── Bangun list of dicts (TANPA iterrows) ─────────────
    result_df = pd.DataFrame({
        'id':                  cust_id.values,
        'name':                name.values,
        'gender':              gender.values,
        'age':                 age.values,
        'country':             country.values,
        'city':                city.values,
        'customer_segment':    segment.values,
        'contract_type':       contract.values,
        'signup_channel':      signup_channel.values,
        'tenure_months':       tenure.values,
        'monthly_logins':      logins.values,
        'last_login':          last_login.values,
        'monthly_fee':         fee.values,
        'total_revenue':       total_rev.values,
        'payment_method':      payment.values,
        'payment_failures':    failures.values,
        'discount_applied':    discount.values,
        'support_tickets':     tickets.values,
        'complaint_type':      complaint_type.values,
        'avg_resolution_time': res_time.values,
        'escalations':         escl.values,
        'csat_score':          csat.values,
        'nps_score':           nps.values,
        'survey_response':     survey.values,
        'churn':               final_churn.values,
        'churn_known':         churn_known.values,
        'churn_prob':          churn_prob.values,
        'risk':                risk.astype(str).values,
    })

    # Bersihkan NaN/Inf → None sebelum konversi ke dict
    # NaN tidak valid di JSON (harus null), dan Inf juga tidak valid
    result_df = result_df.replace({np.nan: None, np.inf: None, -np.inf: None})

    # Konversi ke list of dicts (to_dict('records') jauh lebih cepat dari loop)
    customers = result_df.to_dict('records')

    update_job(95, "Selesai memproses...")
    print(f"✅ Processed {len(customers)} customers")
    return customers

# ── Background Upload Worker ─────────────────────────────
def run_upload_job(job_id: str, filepath: str, filename: str):
    global customers_data, current_csv_filename
    try:
        upload_jobs[job_id]['status']   = 'processing'
        upload_jobs[job_id]['message']  = 'Membaca file CSV...'
        upload_jobs[job_id]['progress'] = 2

        df = pd.read_csv(filepath)

        customers_data       = process_customer_data(df, job_id=job_id)
        current_csv_filename = filename

        total    = len(customers_data)
        churned  = sum(1 for c in customers_data if c.get('churn') == 1)
        high_risk= sum(1 for c in customers_data if c.get('risk') == 'high')

        upload_jobs[job_id].update({
            'status':           'done',
            'progress':         100,
            'message':          f'Berhasil mengimpor {total} customers',
            'total_customers':  total,
            'churned':          churned,
            'high_risk':        high_risk,
        })

    except Exception as e:
        traceback.print_exc()
        upload_jobs[job_id].update({
            'status':  'error',
            'message': str(e),
            'progress': 0,
        })

# ── Routes ───────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'data_loaded': len(customers_data) > 0,
        'total_customers': len(customers_data),
        'model_ready': model is not None,
        'model_type': model_type_name,
        'model_path': MODEL_PATH,
        'model_file_exists': os.path.exists(MODEL_PATH),
    })

@app.route('/api/upload', methods=['POST'])
def upload_csv():
    """Upload CSV — langsung return job_id, proses di background."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Only CSV files allowed'}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        job_id = str(uuid.uuid4())[:8]
        upload_jobs[job_id] = {
            'status':   'queued',
            'progress': 0,
            'message':  'Upload diterima, menunggu pemrosesan...',
        }

        # Jalankan di background thread — tidak blocking
        t = threading.Thread(target=run_upload_job, args=(job_id, filepath, filename), daemon=True)
        t.start()

        return jsonify({
            'success':  True,
            'job_id':   job_id,
            'message':  'Upload diterima. Polling /api/upload/status/<job_id> untuk progress.',
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/upload/status/<job_id>', methods=['GET'])
def upload_status(job_id):
    """Frontend polling endpoint untuk cek progress upload."""
    if job_id not in upload_jobs:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(upload_jobs[job_id])

@app.route('/api/dashboard', methods=['GET'])
def get_dashboard():
    global customers_data

    data = customers_data

    # ── Filter ────────────────────────────────────────────
    segment = request.args.get('segment', '')
    country = request.args.get('country', '')
    if segment: data = [c for c in data if c.get('customer_segment') == segment]
    if country: data = [c for c in data if c.get('country') == country]

    if not data:
        return jsonify({'empty': True})

    # ── Hitung stats dengan pandas (cepat) ───────────────
    df = pd.DataFrame(data)

    total    = len(df)
    churned  = int((df['churn'] == 1).sum())
    active   = total - churned
    churn_rate = round(churned / total * 100, 1) if total else 0

    high_risk = int((df['risk'] == 'high').sum())
    med_risk  = int((df['risk'] == 'medium').sum())
    low_risk  = int((df['risk'] == 'low').sum())

    avg_csat   = round(df['csat_score'].mean(), 2)
    avg_tenure = round(df['tenure_months'].mean(), 1)
    avg_fee    = round(df['monthly_fee'].mean(), 2)
    total_rev  = round(df['total_revenue'].sum(), 2)

    # NPS
    nps_vals   = df['nps_score']
    promoters  = int((nps_vals >= 9).sum())
    passives   = int(((nps_vals >= 7) & (nps_vals < 9)).sum())
    detractors = int((nps_vals < 7).sum())
    nps_score  = round(((promoters - detractors) / total * 100), 1) if total else 0

    # MRR
    total_mrr  = round(df['monthly_fee'].sum(), 2)
    lost_mrr   = round(df.loc[df['churn'] == 1, 'monthly_fee'].sum(), 2)
    at_risk_mrr= round(df.loc[df['risk'] == 'high', 'monthly_fee'].sum(), 2)
    arpu       = round(total_mrr / active, 2) if active else 0

    # Segment stats
    seg_stats = []
    for seg_name, grp in df.groupby('customer_segment'):
        ch = (grp['churn'] == 1).sum()
        seg_stats.append({
            'segment': seg_name,
            'total': len(grp),
            'churned': int(ch),
            'churn_rate': round(ch / len(grp) * 100, 1),
            'avg_csat': round(grp['csat_score'].mean(), 2),
            'avg_fee': round(grp['monthly_fee'].mean(), 2),
        })

    # Country stats (top 10)
    country_stats = []
    for cname, grp in df.groupby('country'):
        ch = (grp['churn'] == 1).sum()
        country_stats.append({
            'country': cname,
            'total': len(grp),
            'churned': int(ch),
            'churn_rate': round(ch / len(grp) * 100, 1),
        })
    country_stats = sorted(country_stats, key=lambda x: x['total'], reverse=True)[:10]

    # Contract stats
    contract_stats = []
    for ctype, grp in df.groupby('contract_type'):
        ch = (grp['churn'] == 1).sum()
        contract_stats.append({
            'contract_type': ctype, 'total': len(grp), 'churned': int(ch),
            'churn_rate': round(ch / len(grp) * 100, 1),
        })

    # Channel stats
    channel_stats = []
    for ch_name, grp in df.groupby('signup_channel'):
        ch = (grp['churn'] == 1).sum()
        channel_stats.append({
            'channel': ch_name, 'total': len(grp), 'churned': int(ch),
            'churn_rate': round(ch / len(grp) * 100, 1),
        })

    # Churn reasons
    reasons_raw = df.loc[df['churn'] == 1, 'complaint_type'].value_counts()
    churn_reasons = [{'reason': r, 'count': int(c)} for r, c in reasons_raw.items() if r != 'None'][:8]

    # Churn by tenure
    bins   = [0, 6, 12, 24, 36, 9999]
    labels = ['0-6m', '6-12m', '12-24m', '24-36m', '36m+']
    df['tenure_band'] = pd.cut(df['tenure_months'], bins=bins, labels=labels, right=True)
    churn_by_tenure = []
    for band, grp in df.groupby('tenure_band', observed=True):
        ch = (grp['churn'] == 1).sum()
        churn_by_tenure.append({
            'range': str(band), 'total': len(grp),
            'churn_rate': round(ch / len(grp) * 100, 1) if len(grp) else 0,
        })

    # Payment failure churn
    pf_stats = []
    for f in range(5):
        grp = df[df['payment_failures'] == f] if f < 4 else df[df['payment_failures'] >= 4]
        ch  = (grp['churn'] == 1).sum()
        pf_stats.append({
            'failures': str(f) if f < 4 else '4+',
            'churn_rate': round(ch / len(grp) * 100, 1) if len(grp) else 0,
        })

    # CSAT distribution
    csat_dist = []
    for s in [1, 2, 3, 4, 5]:
        cnt = int((df['csat_score'].round() == s).sum())
        csat_dist.append({'score': s, 'count': cnt, 'pct': round(cnt / total * 100, 1) if total else 0})

    # High-risk alert list (top 10)
    alert_df   = df[df['risk'] == 'high'].nlargest(10, 'churn_prob')
    alert_list = alert_df[['id', 'name', 'customer_segment', 'country',
                            'support_tickets', 'last_login', 'csat_score',
                            'churn_prob', 'risk']].to_dict('records')

    return jsonify({
        'empty': False,
        'total_customers': total,
        'active_customers': active,
        'churned_customers': churned,
        'churn_rate': churn_rate,
        'high_risk_customers': high_risk,
        'medium_risk_customers': med_risk,
        'low_risk_customers': low_risk,
        'avg_csat': avg_csat,
        'avg_tenure': avg_tenure,
        'nps_score': nps_score,
        'nps_distribution': {'promoters': promoters, 'passives': passives, 'detractors': detractors},
        'total_revenue': total_rev,
        'avg_monthly_fee': avg_fee,
        'mrr': {
            'total': total_mrr,
            'lost_90d': lost_mrr,
            'at_risk': at_risk_mrr,
            'arpu': arpu,
        },
        'segment_stats': seg_stats,
        'country_stats': country_stats,
        'contract_stats': contract_stats,
        'channel_stats': channel_stats,
        'churn_reasons': churn_reasons,
        'churn_by_tenure': churn_by_tenure,
        'payment_failure_stats': pf_stats,
        'csat_distribution': csat_dist,
        'alert_list': alert_list,
    })

@app.route('/api/customers', methods=['GET'])
def get_customers():
    global customers_data

    if not customers_data:
        return jsonify({'data': [], 'total': 0, 'total_pages': 1, 'page': 1})

    segment  = request.args.get('segment', '')
    risk     = request.args.get('risk', '')
    country  = request.args.get('country', '')
    churn    = request.args.get('churn', '')
    search   = request.args.get('search', '').lower()
    page     = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 15))

    filtered = customers_data
    if segment: filtered = [c for c in filtered if c.get('customer_segment') == segment]
    if risk:    filtered = [c for c in filtered if c.get('risk') == risk]
    if country: filtered = [c for c in filtered if c.get('country') == country]
    if churn:   filtered = [c for c in filtered if str(c.get('churn', 0)) == churn]
    if search:
        filtered = [c for c in filtered if
                    search in c.get('name', '').lower() or
                    search in str(c.get('id', '')).lower() or
                    search in c.get('country', '').lower()]

    total     = len(filtered)
    start     = (page - 1) * per_page
    paginated = filtered[start:start + per_page]

    return jsonify({
        'data': paginated, 'total': total, 'page': page,
        'per_page': per_page,
        'total_pages': max(1, (total + per_page - 1) // per_page),
    })

@app.route('/api/predict', methods=['POST'])
def predict_single():
    try:
        data     = request.json or {}
        customer = {
            'csat_score':          float(data.get('csat_score', 3.5)),
            'payment_failures':    int(data.get('payment_failures', 0)),
            'tenure_months':       int(data.get('tenure_months', 12)),
            'monthly_logins':      int(data.get('monthly_logins', 10)),
            'total_revenue':       float(data.get('total_revenue', 0)),
            'discount_applied':    int(data.get('discount_applied', 0)),
            'survey_response':     int(data.get('survey_response', 0)),
            'complaint_type':      data.get('complaint_type', 'None'),
            'signup_channel':      data.get('signup_channel', 'Organic'),
            'contract_type':       data.get('contract_type', 'Monthly'),
            'customer_segment':    data.get('customer_segment', 'Standard'),
            'nps_score':           int(data.get('nps_score', 0)),
            'escalations':         int(data.get('escalations', 0)),
            'avg_resolution_time': float(data.get('avg_resolution_time', 0)),
            'support_tickets':     int(data.get('support_tickets', 0)),
            'last_login':          int(data.get('last_login', 30)),
            'monthly_fee':         float(data.get('monthly_fee', 100)),
        }
        # Single predict menggunakan batch dengan 1 baris
        single_df = pd.DataFrame([customer])
        proba_s, risk_s = predict_batch(single_df)
        prob = float(proba_s.iloc[0])
        risk = str(risk_s.iloc[0])

        return jsonify({
            'success': True,
            'probability': round(prob * 100, 1),
            'risk': risk,
            'prediction': 'Churn' if prob > 0.5 else 'Active',
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/model/info', methods=['GET'])
def model_info():
    global model
    info = {
        'model_type': model_type_name,
        'is_loaded': model is not None,
        'model_path': MODEL_PATH,
        'model_file_exists': os.path.exists(MODEL_PATH),
        'feature_mapping': list(FEATURE_MAPPING.keys()),
        'total_features': len(FEATURE_MAPPING),
        'total_customers': len(customers_data),
    }
    if model is not None and hasattr(model, 'feature_importances_'):
        fi = model.feature_importances_
        feat_names = list(FEATURE_MAPPING.keys())
        info['feature_importances'] = [
            {'feature': feat_names[i] if i < len(feat_names) else f'f{i}', 'importance': round(float(v), 4)}
            for i, v in enumerate(fi)
        ]
    elif model is not None and hasattr(model, 'coef_'):
        coefs = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
        feat_names = list(FEATURE_MAPPING.keys())
        info['feature_importances'] = [
            {'feature': feat_names[i] if i < len(feat_names) else f'f{i}', 'importance': round(abs(float(v)), 4)}
            for i, v in enumerate(coefs)
        ]
    return jsonify(info)

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """
    Endpoint utama yang dipanggil frontend.
    Mengembalikan data dalam format DICT (bukan array) sesuai kebutuhan index.html.
    """
    global customers_data

    data = customers_data

    # ── Filter ────────────────────────────────────────────
    segment_f = request.args.get('segment', '')
    country_f = request.args.get('country', '')
    if segment_f and segment_f != 'all':
        data = [c for c in data if c.get('customer_segment') == segment_f]
    if country_f and country_f != 'all':
        data = [c for c in data if c.get('country') == country_f]

    if not data:
        return jsonify({'empty': True})

    df = pd.DataFrame(data)

    total     = len(df)
    churned   = int((df['churn'] == 1).sum())
    active    = total - churned
    churn_rate = round(churned / total * 100, 1) if total else 0

    high_risk = int((df['risk'] == 'high').sum())
    med_risk  = int((df['risk'] == 'medium').sum())
    low_risk  = int((df['risk'] == 'low').sum())

    avg_csat   = round(float(df['csat_score'].mean()), 2)
    avg_tenure = round(float(df['tenure_months'].mean()), 1)
    avg_fee    = round(float(df['monthly_fee'].mean()), 2)
    total_rev  = round(float(df['total_revenue'].sum()), 2)

    # NPS
    nps_vals   = df['nps_score']
    promoters  = int((nps_vals >= 9).sum())
    passives   = int(((nps_vals >= 7) & (nps_vals < 9)).sum())
    detractors = int((nps_vals < 7).sum())
    nps_score  = round((promoters - detractors) / total * 100, 1) if total else 0

    # MRR
    total_mrr   = round(float(df['monthly_fee'].sum()), 2)
    lost_mrr    = round(float(df.loc[df['churn'] == 1, 'monthly_fee'].sum()), 2)
    at_risk_mrr = round(float(df.loc[df['risk'] == 'high', 'monthly_fee'].sum()), 2)
    arpu        = round(total_mrr / active, 2) if active else 0

    # ── segment_stats sebagai DICT (frontend expects this) ──
    segment_stats = {}
    for seg_name, grp in df.groupby('customer_segment'):
        ch       = int((grp['churn'] == 1).sum())
        hr       = int((grp['risk'] == 'high').sum())
        rev      = round(float(grp['total_revenue'].sum()), 2)
        segment_stats[seg_name] = {
            'total':      len(grp),
            'churned':    ch,
            'churn_rate': round(ch / len(grp) * 100, 1),
            'high_risk':  hr,
            'revenue':    rev,
            'avg_csat':   round(float(grp['csat_score'].mean()), 2),
            'avg_fee':    round(float(grp['monthly_fee'].mean()), 2),
        }

    # ── country_stats sebagai DICT ──
    country_stats = {}
    for cname, grp in df.groupby('country'):
        ch = int((grp['churn'] == 1).sum())
        country_stats[cname] = {
            'total':      len(grp),
            'churned':    ch,
            'churn_rate': round(ch / len(grp) * 100, 1),
        }

    # ── contract_stats sebagai DICT ──
    contract_stats = {}
    for ctype, grp in df.groupby('contract_type'):
        ch = int((grp['churn'] == 1).sum())
        contract_stats[ctype] = {
            'total':      len(grp),
            'churned':    ch,
            'churn_rate': round(ch / len(grp) * 100, 1),
        }

    # ── channel_stats sebagai DICT ──
    channel_stats = {}
    for ch_name, grp in df.groupby('signup_channel'):
        ch = int((grp['churn'] == 1).sum())
        channel_stats[ch_name] = {
            'total':      len(grp),
            'churned':    ch,
            'churn_rate': round(ch / len(grp) * 100, 1),
        }

    # ── churn_reasons sebagai DICT {reason: count} ──
    reasons_raw  = df.loc[df['churn'] == 1, 'complaint_type'].value_counts()
    churn_reasons = {r: int(c) for r, c in reasons_raw.items() if r != 'None'}

    # ── churn_by_tenure (array — sudah benar) ──
    bins   = [0, 6, 12, 24, 36, 9999]
    labels = ['0-6m', '6-12m', '12-24m', '24-36m', '36m+']
    df2 = df.copy()
    df2['tenure_band'] = pd.cut(df2['tenure_months'], bins=bins, labels=labels, right=True)
    churn_by_tenure = []
    for band, grp in df2.groupby('tenure_band', observed=True):
        ch = int((grp['churn'] == 1).sum())
        churn_by_tenure.append({
            'range': str(band), 'total': len(grp),
            'churn_rate': round(ch / len(grp) * 100, 1) if len(grp) else 0,
        })

    # ── payment_failure_stats (array) ──
    pf_stats = []
    for f in range(5):
        grp = df[df['payment_failures'] == f] if f < 4 else df[df['payment_failures'] >= 4]
        ch  = int((grp['churn'] == 1).sum())
        pf_stats.append({
            'failures':   str(f) if f < 4 else '4+',
            'churn_rate': round(ch / len(grp) * 100, 1) if len(grp) else 0,
        })

    # ── csat_distribution (array) ──
    csat_dist = []
    for s in [1, 2, 3, 4, 5]:
        cnt = int((df['csat_score'].round() == s).sum())
        csat_dist.append({'score': s, 'count': cnt, 'pct': round(cnt / total * 100, 1) if total else 0})

    # ── alert_list (top 10 high risk) ──
    alert_df   = df[df['risk'] == 'high'].nlargest(10, 'churn_prob')
    alert_list = alert_df[['id', 'name', 'customer_segment', 'country',
                            'support_tickets', 'last_login', 'csat_score',
                            'churn_prob', 'risk']].to_dict('records')

    return jsonify({
        'empty':                False,
        'total_customers':      total,
        'active_customers':     active,
        'churned_customers':    churned,
        'churn_rate':           churn_rate,
        'high_risk_customers':  high_risk,
        'medium_risk_customers': med_risk,
        'low_risk_customers':   low_risk,
        'avg_csat':             avg_csat,
        'avg_tenure':           avg_tenure,
        'nps_score':            nps_score,
        'nps_distribution':     {'promoters': promoters, 'passives': passives, 'detractors': detractors},
        'total_revenue':        total_rev,
        'avg_monthly_fee':      avg_fee,
        'mrr': {
            'total':    total_mrr,
            'lost_90d': lost_mrr,
            'at_risk':  at_risk_mrr,
            'arpu':     arpu,
        },
        # ↓ Format DICT — sesuai yang diharapkan frontend
        'segment_stats':          segment_stats,
        'country_stats':          country_stats,
        'contract_stats':         contract_stats,
        'channel_stats':          channel_stats,
        'churn_reasons':          churn_reasons,
        # ↓ Format array — sudah benar
        'churn_by_tenure':        churn_by_tenure,
        'payment_failure_stats':  pf_stats,
        'csat_distribution':      csat_dist,
        'alert_list':             alert_list,
    })


@app.route('/api/export', methods=['GET'])
def export_csv():
    global customers_data
    if not customers_data:
        return jsonify({'error': 'No data to export'}), 404
    df     = pd.DataFrame(customers_data)
    from io import StringIO
    output = StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return send_file(
        output, mimetype='text/csv', as_attachment=True,
        download_name=f'churn_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    )

# ── Main ─────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*65)
    print("🚀 ChurnPredict — Customer Churn Dashboard (OPTIMIZED)")
    print("="*65)
    print(f"📊 Model     : {model_type_name}")
    print(f"💾 Model path: {MODEL_PATH} ({'found ✅' if os.path.exists(MODEL_PATH) else 'NOT FOUND — using rule-based ⚠️'})")
    print(f"\n📋 Feature mapping ({len(FEATURE_MAPPING)} features):")
    for k in FEATURE_MAPPING:
        print(f"   • {k}")
    print(f"\n📁 Pastikan index.html ada di folder: templates/")
    print(f"📍 Open http://localhost:5000")
    print("="*65 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)