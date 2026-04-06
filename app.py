"""
===============================================================================
  BURIRAM TOURISM FORECASTING DASHBOARD
  แนวคิด CRISP-DM 6 ขั้นตอน
  ข้อมูล: นักท่องเที่ยวจังหวัดบุรีรัมย์ ปี 2556–2568 (พ.ศ.)
===============================================================================
"""
import thai_support
thai_support.setup_thai_font()
thai_support.setup_pandas_display()
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Buriram Tourism Dashboard",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2rem; font-weight: 800;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        padding: 0.5rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px; padding: 1rem 1.2rem;
        color: white; text-align: center;
    }
    .metric-card .val { font-size: 1.8rem; font-weight: 700; }
    .metric-card .lbl { font-size: 0.85rem; opacity: 0.9; }
    .section-header {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        color: white; padding: 0.5rem 1rem; border-radius: 8px;
        font-weight: 700; margin-bottom: 1rem;
    }
    .crisp-badge {
        display: inline-block;
        background: #0f3460; color: white;
        border-radius: 20px; padding: 2px 12px;
        font-size: 0.8rem; font-weight: 600; margin-right: 6px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #f0f2f6; border-radius: 8px 8px 0 0;
        padding: 8px 16px; font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
THAI_MONTHS = {
    'มกราคม': 1, 'กุมภาพันธ์': 2, 'มีนาคม': 3, 'เมษายน': 4,
    'พฤษภาคม': 5, 'มิถุนายน': 6, 'กรกฎาคม': 7, 'สิงหาคม': 8,
    'กันยายน': 9, 'ตุลาคม': 10, 'พฤศจิกายน': 11, 'ธันวาคม': 12,
}
MONTHLY_MONTHS = list(THAI_MONTHS.keys())
QUARTERLY_LABELS = ['มกราคม - มีนาคม', 'เมษายน - มิถุนายน', 'กรกฎาคม - กันยายน', 'ตุลาคม - ธันวาคม']
QUARTER_MAP = {
    'มกราคม - มีนาคม': 1, 'เมษายน - มิถุนายน': 2,
    'กรกฎาคม - กันยายน': 3, 'ตุลาคม - ธันวาคม': 4,
}
EVENT_COLS = ['MotoGP', 'Covid', 'Marathon', 'PhanomRung_Festival']
EVENT_LABELS = {'MotoGP': '🏍️ MotoGP', 'Covid': '😷 COVID-19', 'Marathon': '🏃 Marathon', 'PhanomRung_Festival': '🏯 ผานรุ้ง'}
COLORS = ['#667eea', '#f5576c', '#4facfe', '#43e97b', '#fa709a', '#fee140', '#a18cd1', '#fda085']

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & CLEANING  (CRISP-DM ขั้น 1-3)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_clean(path: str):
    """CRISP-DM Phase 2-3: Data Understanding & Preparation"""
    df_raw = pd.read_csv(path)

    # ── Clean numeric columns with commas ──
    num_cols_str = ['Guests_total', 'Total_vis', 'Thai_vis', 'Foreign_vis']
    for c in num_cols_str:
        df_raw[c] = df_raw[c].astype(str).str.replace(',', '', regex=False)
        df_raw[c] = pd.to_numeric(df_raw[c], errors='coerce')

    # ── Separate monthly vs quarterly rows ──
    df_monthly = df_raw[df_raw['Month&Quarter'].isin(MONTHLY_MONTHS)].copy()
    df_quarterly = df_raw[df_raw['Month&Quarter'].isin(QUARTERLY_LABELS)].copy()

    # ── Monthly: keep rows that have tourism data ──
    df_monthly = df_monthly.dropna(subset=['Total_vis']).copy()
    df_monthly['Month_num'] = df_monthly['Month&Quarter'].map(THAI_MONTHS)
    df_monthly = df_monthly.sort_values(['Year', 'Month_num']).reset_index(drop=True)

    # ── Quarterly: keep rows that have tourism data ──
    df_quarterly = df_quarterly.dropna(subset=['Total_vis']).copy()
    df_quarterly['Quarter'] = df_quarterly['Month&Quarter'].map(QUARTER_MAP)
    for ec in EVENT_COLS:
        df_quarterly[ec] = df_quarterly[ec].fillna(0).astype(int)
    df_quarterly = df_quarterly.sort_values(['Year', 'Quarter']).reset_index(drop=True)

    # ── Annual aggregation from quarterly ──
    df_annual = (
        df_quarterly.groupby('Year')
        .agg(
            Total_vis=('Total_vis', 'sum'),
            MotoGP=('MotoGP', 'max'),
            Covid=('Covid', 'max'),
            Marathon=('Marathon', 'max'),
            PhanomRung_Festival=('PhanomRung_Festival', 'max'),
        )
        .reset_index()
    )

    # ── Clean log for display ──
    cleaning_notes = [
        "✅ ลบ comma ออกจาก numeric columns (Total_vis, Guests_total, etc.)",
        "✅ แยกข้อมูลรายเดือน (monthly) กับรายไตรมาส (quarterly) ออกจากกัน",
        "✅ ลบแถวที่ไม่มีข้อมูล Total_vis (แถวฟุตบอลที่ซ้ำกัน)",
        f"✅ ข้อมูลรายไตรมาส: {len(df_quarterly)} แถว, ปี 2556–2568",
        f"✅ ข้อมูลรายเดือน: {len(df_monthly)} แถว, ปี 2567–2568",
        f"✅ ข้อมูลรายปี (aggregate): {len(df_annual)} ปี",
        "✅ Event flags: MotoGP, Covid, Marathon, PhanomRung_Festival",
    ]

    return df_raw, df_quarterly, df_monthly, df_annual, cleaning_notes


@st.cache_data
def build_ml_features(df_quarterly: pd.DataFrame):
    """CRISP-DM Phase 4: Modeling — feature engineering"""
    df = df_quarterly.copy()

    # Lag features
    df = df.sort_values(['Year', 'Quarter']).reset_index(drop=True)
    df['vis_lag1'] = df['Total_vis'].shift(1)
    df['vis_lag2'] = df['Total_vis'].shift(2)
    df['vis_lag4'] = df['Total_vis'].shift(4)   # same quarter last year

    # Quarter dummies
    for q in range(1, 5):
        df[f'Q{q}'] = (df['Quarter'] == q).astype(int)

    # Year trend (normalized)
    df['year_trend'] = df['Year'] - df['Year'].min()

    # Drop rows with NaN lags
    df = df.dropna(subset=['vis_lag1', 'vis_lag2', 'vis_lag4']).reset_index(drop=True)

    feature_cols = ['vis_lag1', 'vis_lag2', 'vis_lag4',
                    'Q1', 'Q2', 'Q3', 'Q4',
                    'year_trend', 'MotoGP', 'Covid', 'Marathon', 'PhanomRung_Festival']

    return df, feature_cols


@st.cache_data
def train_and_evaluate(df_feat: pd.DataFrame, feature_cols: list):
    """CRISP-DM Phase 4-5: Modeling & Evaluation"""
    X = df_feat[feature_cols].values
    y = df_feat['Total_vis'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)

        # Cross-val R² (leave some out)
        cv_r2 = cross_val_score(model, X_scaled, y, cv=min(5, len(y)//4), scoring='r2')

        results[name] = {
            'model': model,
            'scaler': scaler,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'CV_R2_mean': cv_r2.mean(),
            'y_pred': y_pred,
        }

    # Best model = highest R²
    best_name = max(results, key=lambda k: results[k]['R2'])
    return results, best_name, scaler, feature_cols


def predict_2569(df_feat: pd.DataFrame, results: dict, best_name: str,
                 scaler, feature_cols: list,
                 events_2569: dict):
    """Predict year 2569 (all 4 quarters) using the best model"""
    model = results[best_name]['model']
    last_year = df_feat['Year'].max()  # should be 2568

    preds = []
    # We use the last available lags from df_feat for the first quarter
    last_row = df_feat.iloc[-1]
    lag1 = last_row['Total_vis']
    lag2 = last_row['vis_lag1']
    lag4_base = df_feat[df_feat['Year'] == last_year - 1]['Total_vis'].values

    # Annual totals for same-quarter-last-year reference
    prev_year_q = df_feat[df_feat['Year'] == last_year].set_index('Quarter')['Total_vis'].to_dict()

    running_history = df_feat[['Year', 'Quarter', 'Total_vis']].copy()

    for q in range(1, 5):
        q_feats = {f'Q{i}': 1 if i == q else 0 for i in range(1, 5)}
        lag4 = prev_year_q.get(q, lag1)

        row = {
            'vis_lag1': lag1,
            'vis_lag2': lag2,
            'vis_lag4': lag4,
            **q_feats,
            'year_trend': last_year + 1 - df_feat['Year'].min(),
            'MotoGP': events_2569.get('MotoGP', 0),
            'Covid': events_2569.get('Covid', 0),
            'Marathon': events_2569.get('Marathon', 0),
            'PhanomRung_Festival': events_2569.get('PhanomRung_Festival', 0),
        }

        X_new = np.array([[row[f] for f in feature_cols]])
        X_new_scaled = scaler.transform(X_new)
        pred = model.predict(X_new_scaled)[0]
        preds.append({'Quarter': q, 'Predicted_vis': max(0, pred)})

        lag2 = lag1
        lag1 = pred

    return pd.DataFrame(preds)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏟️ Buriram Tourism")
    st.markdown("**CRISP-DM Dashboard**")
    st.divider()

    uploaded = st.file_uploader("📂 อัพโหลดไฟล์ CSV", type=["csv"])
    st.caption("หากไม่ได้อัพโหลด จะใช้ไฟล์ตัวอย่างที่ฝังมาแล้ว")
    st.divider()

    st.markdown("### ⚙️ ตั้งค่า Event ปี 2569")
    ev_motogp = st.toggle("🏍️ MotoGP", value=True)
    ev_covid = st.toggle("😷 COVID-19", value=False)
    ev_marathon = st.toggle("🏃 Marathon", value=True)
    ev_phanomrung = st.toggle("🏯 PhanomRung Festival", value=True)
    events_2569 = {
        'MotoGP': int(ev_motogp),
        'Covid': int(ev_covid),
        'Marathon': int(ev_marathon),
        'PhanomRung_Festival': int(ev_phanomrung),
    }

    st.divider()
    st.markdown("### 📊 Model Selection")
    manual_model = st.selectbox(
        "เลือกโมเดลเอง (หรือใช้ Auto-Select)",
        ['Auto-Select (Best R²)', 'Linear Regression', 'Ridge Regression', 'Random Forest', 'Gradient Boosting'],
    )

    st.divider()
    st.markdown("""
    <small>
    🎓 CRISP-DM 6 ขั้นตอน<br>
    1️⃣ Business Understanding<br>
    2️⃣ Data Understanding<br>
    3️⃣ Data Preparation<br>
    4️⃣ Modeling<br>
    5️⃣ Evaluation<br>
    6️⃣ Deployment
    </small>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH = "dataCI02-09-03-2569.csv"
if uploaded:
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded.read())
        DATA_PATH = tmp.name

try:
    df_raw, df_quarterly, df_monthly, df_annual, cleaning_notes = load_and_clean(DATA_PATH)
    df_feat, feature_cols = build_ml_features(df_quarterly)
    results, best_name, scaler, feature_cols = train_and_evaluate(df_feat, feature_cols)
    pred_2569 = predict_2569(df_feat, results, best_name, scaler, feature_cols, events_2569)

    selected_model = best_name if manual_model == 'Auto-Select (Best R²)' else manual_model.replace('Random Forest', 'Random Forest').replace('Gradient Boosting', 'Gradient Boosting')
    # fix mapping
    model_name_map = {
        'Linear Regression': 'Linear Regression',
        'Ridge Regression': 'Ridge Regression',
        'Random Forest': 'Random Forest',
        'Gradient Boosting': 'Gradient Boosting',
    }
    if manual_model != 'Auto-Select (Best R²)':
        for k in model_name_map:
            if k in manual_model:
                selected_model = k
                break

    DATA_OK = True
except Exception as e:
    st.error(f"❌ ไม่สามารถโหลดข้อมูลได้: {e}")
    st.info("กรุณาอัพโหลดไฟล์ dataCI02-09-03-2569.csv ผ่าน Sidebar")
    DATA_OK = False
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">🏏 Buriram Tourism Forecasting Dashboard</h1>', unsafe_allow_html=True)
st.markdown("**ระบบวิเคราะห์และพยากรณ์นักท่องเที่ยวจังหวัดบุรีรัมย์** | แนวคิด CRISP-DM | ข้อมูลปี 2556–2568 (พ.ศ.)")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# KPI METRICS ROW
# ─────────────────────────────────────────────────────────────────────────────
latest_year = df_annual['Year'].max()
latest_total = int(df_annual[df_annual['Year'] == latest_year]['Total_vis'].values[0])
prev_total = int(df_annual[df_annual['Year'] == latest_year - 1]['Total_vis'].values[0])
growth = (latest_total - prev_total) / prev_total * 100
pred_total_2569 = int(pred_2569['Predicted_vis'].sum())
best_r2 = results[best_name]['R2']

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="val">{latest_total:,.0f}</div>
        <div class="lbl">นักท่องเที่ยวปี {latest_year} (พ.ศ.)</div>
    </div>""", unsafe_allow_html=True)
with c2:
    arrow = "📈" if growth >= 0 else "📉"
    st.markdown(f"""
    <div class="metric-card" style="background:linear-gradient(135deg,#f093fb,#f5576c)">
        <div class="val">{arrow} {growth:+.1f}%</div>
        <div class="lbl">การเติบโต YoY</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="metric-card" style="background:linear-gradient(135deg,#4facfe,#00f2fe)">
        <div class="val">{pred_total_2569:,.0f}</div>
        <div class="lbl">พยากรณ์ปี 2569 (รวม)</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div class="metric-card" style="background:linear-gradient(135deg,#43e97b,#38f9d7)">
        <div class="val">{best_r2:.3f}</div>
        <div class="lbl">Best Model R² ({best_name[:10]})</div>
    </div>""", unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 CRISP-DM Overview",
    "🔍 Data Exploration",
    "📅 Monthly Analysis",
    "🤖 Model & Evaluation",
    "🔮 Forecast 2569",
    "🎯 Event Impact",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 : CRISP-DM OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">🔄 CRISP-DM Framework — 6 ขั้นตอน</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        phases = [
            ("1️⃣ Business Understanding",
             "ทำความเข้าใจเป้าหมายทางธุรกิจ: พยากรณ์จำนวนนักท่องเที่ยวจังหวัดบุรีรัมย์ เพื่อวางแผนด้านการท่องเที่ยวและประเมินผลกระทบจาก Event ต่างๆ"),
            ("2️⃣ Data Understanding",
             f"ข้อมูลรายไตรมาส {len(df_quarterly)} แถว | รายเดือน {len(df_monthly)} แถว | ปี 2556–2568 | มี Event flags: MotoGP, COVID, Marathon, ผานรุ้ง"),
            ("3️⃣ Data Preparation",
             "ลบ comma จาก numeric, แยก monthly/quarterly, สร้าง lag features (lag1, lag2, lag4), quarter dummies, year trend"),
            ("4️⃣ Modeling",
             "เปรียบเทียบ 4 โมเดล: Linear Regression, Ridge Regression, Random Forest, Gradient Boosting"),
            ("5️⃣ Evaluation",
             f"ประเมินด้วย MAE, RMSE, R² | Best model: **{best_name}** (R²={best_r2:.4f})"),
            ("6️⃣ Deployment",
             "Deploy บน Streamlit Cloud ผ่าน GitHub — Dashboard สำหรับผู้ใช้งานทั่วไปดูการพยากรณ์และวิเคราะห์ข้อมูล"),
        ]
        for title, desc in phases:
            with st.expander(title, expanded=False):
                st.write(desc)

    with col_b:
        st.markdown("#### 📊 Data Cleaning Notes")
        for note in cleaning_notes:
            st.markdown(f"- {note}")
        st.divider()
        st.markdown("#### 📁 Raw Data Preview (10 แถวแรก)")
        st.dataframe(df_raw.head(10), use_container_width=True, height=280)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 : DATA EXPLORATION
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">🔍 Data Exploration — แนวโน้มรายปีและรายไตรมาส</div>', unsafe_allow_html=True)

    # ── Annual trend ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#f8f9fa')

    years = df_annual['Year'].values
    totals = df_annual['Total_vis'].values / 1e6

    ax = axes[0]
    bars = ax.bar(years, totals, color=COLORS[:len(years)], edgecolor='white', linewidth=0.8)
    ax.set_title('จำนวนนักท่องเที่ยวรายปี (ล้านคน)', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('ปี (พ.ศ.)', fontsize=10)
    ax.set_ylabel('จำนวน (ล้านคน)', fontsize=10)
    ax.set_facecolor('#fafafa')
    for bar, val in zip(bars, totals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}M', ha='center', va='bottom', fontsize=8, fontweight='bold')
    # Annotate COVID
    covid_years = df_annual[df_annual['Covid'] == 1]['Year'].values
    for yr in covid_years:
        idx = list(years).index(yr)
        ax.axvspan(yr - 0.5, yr + 0.5, alpha=0.15, color='red')
    ax.tick_params(axis='x', rotation=45)

    # ── Quarterly heatmap ──
    ax2 = axes[1]
    pivot = df_quarterly.pivot_table(index='Year', columns='Quarter', values='Total_vis', aggfunc='sum')
    im = ax2.imshow(pivot.values / 1e5, aspect='auto', cmap='YlOrRd')
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
    ax2.set_yticks(range(len(pivot.index)))
    ax2.set_yticklabels(pivot.index)
    ax2.set_title('Heatmap: นักท่องเที่ยวรายไตรมาส (แสนคน)', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax2, shrink=0.8)
    for i in range(len(pivot.index)):
        for j in range(4):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax2.text(j, i, f'{val/1e5:.1f}', ha='center', va='center', fontsize=8, color='black')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Thai/Foreign breakdown ──
    st.markdown("#### 🌍 สัดส่วนนักท่องเที่ยวไทย vs ต่างชาติ (รายปี)")
    df_q_sum = df_quarterly.copy()
    has_thai = 'Thai_vis' in df_quarterly.columns and df_quarterly['Thai_vis'].notna().sum() > 0
    has_foreign = 'Foreign_vis' in df_quarterly.columns and df_quarterly['Foreign_vis'].notna().sum() > 0

    if has_thai and has_foreign:
        by_year = df_quarterly.groupby('Year').agg(
            Thai=('Thai_vis', 'sum'), Foreign=('Foreign_vis', 'sum')).reset_index()
        fig2, ax3 = plt.subplots(figsize=(12, 4))
        fig2.patch.set_facecolor('#f8f9fa')
        ax3.bar(by_year['Year'], by_year['Thai'] / 1e6, label='ไทย', color='#667eea')
        ax3.bar(by_year['Year'], by_year['Foreign'] / 1e6, bottom=by_year['Thai'] / 1e6, label='ต่างชาติ', color='#f5576c')
        ax3.set_xlabel('ปี (พ.ศ.)', fontsize=10)
        ax3.set_ylabel('ล้านคน', fontsize=10)
        ax3.legend()
        ax3.tick_params(axis='x', rotation=45)
        ax3.set_facecolor('#fafafa')
        st.pyplot(fig2)
        plt.close()
    else:
        st.info("ไม่มีข้อมูลแยกนักท่องเที่ยวไทย/ต่างชาติรายปีเพียงพอ")

    st.markdown("#### 📋 ตารางข้อมูลรายปี")
    st.dataframe(df_annual.style.format({'Total_vis': '{:,.0f}'}), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 : MONTHLY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">📅 Monthly Analysis — ข้อมูลรายเดือน</div>', unsafe_allow_html=True)

    avail_years = sorted(df_monthly['Year'].unique())
    if len(avail_years) == 0:
        st.warning("ไม่มีข้อมูลรายเดือน")
    else:
        col_sel, _ = st.columns([1, 3])
        with col_sel:
            sel_year = st.selectbox("เลือกปี (พ.ศ.)", avail_years, index=len(avail_years) - 1)

        df_sel = df_monthly[df_monthly['Year'] == sel_year].sort_values('Month_num')

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor('#f8f9fa')

        month_labels = [df_sel[df_sel['Month_num'] == m]['Month&Quarter'].values[0]
                        if m in df_sel['Month_num'].values else '—'
                        for m in df_sel['Month_num'].values]

        # Bar chart
        ax = axes[0]
        vals = df_sel['Total_vis'].values
        month_names = df_sel['Month&Quarter'].values
        colors_bar = [COLORS[i % len(COLORS)] for i in range(len(vals))]
        ax.bar(range(len(vals)), vals / 1e3, color=colors_bar, edgecolor='white')
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(month_names, rotation=45, ha='right', fontsize=9)
        ax.set_title(f'นักท่องเที่ยวรายเดือน ปี {sel_year} (พันคน)', fontsize=12, fontweight='bold')
        ax.set_ylabel('พันคน', fontsize=10)
        ax.set_facecolor('#fafafa')

        # Line chart with events highlighted
        ax2 = axes[1]
        ax2.plot(range(len(vals)), vals / 1e3, 'o-', color='#667eea', linewidth=2, markersize=8)
        ax2.fill_between(range(len(vals)), vals / 1e3, alpha=0.15, color='#667eea')
        for i, row in enumerate(df_sel.itertuples()):
            events_here = [EVENT_LABELS[e] for e in EVENT_COLS if getattr(row, e, 0) == 1]
            if events_here:
                ax2.axvline(i, color='#f5576c', linestyle='--', alpha=0.6)
                ax2.annotate('\n'.join(events_here), xy=(i, vals[i] / 1e3),
                             xytext=(5, 10), textcoords='offset points', fontsize=7, color='#f5576c')
        ax2.set_xticks(range(len(vals)))
        ax2.set_xticklabels(month_names, rotation=45, ha='right', fontsize=9)
        ax2.set_title(f'แนวโน้มรายเดือน ปี {sel_year}', fontsize=12, fontweight='bold')
        ax2.set_ylabel('พันคน', fontsize=10)
        ax2.set_facecolor('#fafafa')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        if len(avail_years) >= 2:
            st.markdown("#### 📊 เปรียบเทียบรายเดือนระหว่างปี 2567 vs 2568")
            fig3, ax3 = plt.subplots(figsize=(12, 5))
            fig3.patch.set_facecolor('#f8f9fa')
            x = np.arange(12)
            width = 0.35
            for i, yr in enumerate(avail_years[-2:]):
                d = df_monthly[df_monthly['Year'] == yr].sort_values('Month_num')
                months_vals = d.set_index('Month_num')['Total_vis'].reindex(range(1, 13)).fillna(0).values
                ax3.bar(x + i * width, months_vals / 1e3, width, label=f'ปี {yr}', color=COLORS[i])
            ax3.set_xticks(x + width / 2)
            ax3.set_xticklabels(list(THAI_MONTHS.keys()), rotation=45, ha='right', fontsize=9)
            ax3.set_ylabel('พันคน')
            ax3.set_title('เปรียบเทียบนักท่องเที่ยวรายเดือน', fontsize=12, fontweight='bold')
            ax3.legend()
            ax3.set_facecolor('#fafafa')
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()

        st.markdown("#### 📋 ตารางข้อมูลรายเดือน")
        st.dataframe(
            df_sel[['Year', 'Month&Quarter', 'Total_vis', 'Thai_vis', 'Foreign_vis',
                    'MotoGP', 'Covid', 'Marathon', 'PhanomRung_Festival']].style.format(
                        {'Total_vis': '{:,.0f}', 'Thai_vis': '{:,.0f}', 'Foreign_vis': '{:,.0f}'}
                    ),
            use_container_width=True
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 : MODEL & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">🤖 Model Comparison & Evaluation — MAE, RMSE, R²</div>', unsafe_allow_html=True)

    # ── Metrics table ──
    metric_rows = []
    for name, res in results.items():
        metric_rows.append({
            'Model': name,
            'MAE': f"{res['MAE']:,.0f}",
            'RMSE': f"{res['RMSE']:,.0f}",
            'R²': f"{res['R2']:.4f}",
            'CV R² (mean)': f"{res['CV_R2_mean']:.4f}",
            'Best?': '⭐ Best' if name == best_name else '',
        })
    df_metrics = pd.DataFrame(metric_rows)
    st.dataframe(df_metrics.set_index('Model'), use_container_width=True)

    # ── Visual comparison ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#f8f9fa')
    model_names = list(results.keys())
    short_names = ['Linear', 'Ridge', 'RF', 'GBR']
    maes = [results[n]['MAE'] / 1e3 for n in model_names]
    rmses = [results[n]['RMSE'] / 1e3 for n in model_names]
    r2s = [results[n]['R2'] for n in model_names]
    bar_colors = ['#f5576c' if n == best_name else '#667eea' for n in model_names]

    for ax, vals, title, ylabel in zip(axes,
                                       [maes, rmses, r2s],
                                       ['MAE (พันคน)', 'RMSE (พันคน)', 'R² Score'],
                                       ['พันคน', 'พันคน', 'R²']):
        bars = ax.bar(short_names, vals, color=bar_colors, edgecolor='white')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_facecolor('#fafafa')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{v:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.suptitle(f'🏆 Best Model: {best_name}', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Actual vs Predicted ──
    st.markdown(f"#### 📉 Actual vs Predicted — {selected_model}")
    res_sel = results[selected_model]
    y_actual = df_feat['Total_vis'].values
    y_pred = res_sel['y_pred']

    fig2, ax = plt.subplots(figsize=(12, 5))
    fig2.patch.set_facecolor('#f8f9fa')
    ax.plot(y_actual / 1e5, 'o-', color='#667eea', label='Actual', linewidth=2)
    ax.plot(y_pred / 1e5, 's--', color='#f5576c', label='Predicted', linewidth=2)
    ax.fill_between(range(len(y_actual)), y_actual / 1e5, y_pred / 1e5,
                    alpha=0.1, color='#fa709a')
    ax.set_title(f'Actual vs Predicted — {selected_model}', fontsize=12, fontweight='bold')
    ax.set_ylabel('นักท่องเที่ยว (แสนคน)')
    ax.set_xlabel('ลำดับข้อมูล (ไตรมาส)')
    ax.legend()
    ax.set_facecolor('#fafafa')
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # ── Feature Importance (RF/GBR) ──
    if selected_model in ['Random Forest', 'Gradient Boosting']:
        fi = results[selected_model]['model'].feature_importances_
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        fig3.patch.set_facecolor('#f8f9fa')
        sorted_idx = np.argsort(fi)[::-1]
        ax3.bar(range(len(fi)), fi[sorted_idx], color=COLORS[:len(fi)])
        ax3.set_xticks(range(len(fi)))
        ax3.set_xticklabels([feature_cols[i] for i in sorted_idx], rotation=45, ha='right')
        ax3.set_title(f'Feature Importance — {selected_model}', fontsize=12, fontweight='bold')
        ax3.set_facecolor('#fafafa')
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 : FORECAST 2569
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">🔮 Forecast 2569 — พยากรณ์ปีล่าสุด</div>', unsafe_allow_html=True)

    col_info, col_events = st.columns([2, 1])
    with col_info:
        st.markdown(f"**โมเดลที่ใช้:** `{best_name}` (R²={best_r2:.4f})")
        st.markdown("ใช้ข้อมูลปี 2556–2568 (ทั้งหมด) ในการ train แล้วพยากรณ์ปี 2569 ทั้ง 4 ไตรมาส")
    with col_events:
        st.markdown("**Event ปี 2569 ที่เลือก:**")
        for ev_key, ev_val in events_2569.items():
            icon = "✅" if ev_val else "⬜"
            st.markdown(f"{icon} {EVENT_LABELS[ev_key]}")

    # Recalculate with current events selection
    pred_2569 = predict_2569(df_feat, results, best_name, scaler, feature_cols, events_2569)
    pred_2569['Quarter_label'] = pred_2569['Quarter'].map({
        1: 'Q1 (ม.ค.-มี.ค.)', 2: 'Q2 (เม.ย.-มิ.ย.)',
        3: 'Q3 (ก.ค.-ก.ย.)', 4: 'Q4 (ต.ค.-ธ.ค.)'
    })

    total_pred = int(pred_2569['Predicted_vis'].sum())
    total_2568 = int(df_annual[df_annual['Year'] == 2568]['Total_vis'].values[0])
    growth_pred = (total_pred - total_2568) / total_2568 * 100

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#667eea,#764ba2);border-radius:12px;padding:1.2rem;color:white;text-align:center;margin-bottom:1rem;">
        <h2 style="margin:0;color:white;">🎯 พยากรณ์นักท่องเที่ยวปี 2569: {total_pred:,.0f} คน</h2>
        <p style="margin:0.3rem 0 0;">เติบโต {growth_pred:+.1f}% จากปี 2568 ({total_2568:,} คน)</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Historical + Forecast combined chart ──
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor('#f8f9fa')

    # Left: Annual with 2569 forecast
    ax1 = axes[0]
    hist_years = df_annual['Year'].values
    hist_vals = df_annual['Total_vis'].values / 1e6
    ax1.bar(hist_years, hist_vals, color='#667eea', label='ข้อมูลจริง', edgecolor='white')
    ax1.bar([2569], [total_pred / 1e6], color='#f5576c', label='พยากรณ์ 2569',
            edgecolor='white', hatch='//')
    ax1.set_xlabel('ปี (พ.ศ.)', fontsize=10)
    ax1.set_ylabel('ล้านคน', fontsize=10)
    ax1.set_title('ประวัติ + พยากรณ์ปี 2569 (รายปี)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.set_facecolor('#fafafa')
    ax1.tick_params(axis='x', rotation=45)
    # Annotate forecast bar
    ax1.text(2569, total_pred / 1e6 + 0.03, f'{total_pred / 1e6:.2f}M',
             ha='center', va='bottom', fontsize=9, fontweight='bold', color='#f5576c')

    # Right: 2568 actual Q vs 2569 forecast Q
    ax2 = axes[1]
    q_2568 = df_quarterly[df_quarterly['Year'] == 2568][['Quarter', 'Total_vis']].set_index('Quarter')
    q_labels = ['Q1', 'Q2', 'Q3', 'Q4']
    vals_2568 = [q_2568.loc[q, 'Total_vis'] / 1e3 if q in q_2568.index else 0 for q in range(1, 5)]
    vals_2569 = pred_2569.set_index('Quarter')['Predicted_vis'].reindex(range(1, 5)).fillna(0).values / 1e3
    x = np.arange(4)
    ax2.bar(x - 0.2, vals_2568, 0.4, label='2568 (จริง)', color='#667eea', edgecolor='white')
    ax2.bar(x + 0.2, vals_2569, 0.4, label='2569 (พยากรณ์)', color='#f5576c', edgecolor='white', hatch='//')
    ax2.set_xticks(x)
    ax2.set_xticklabels(q_labels)
    ax2.set_ylabel('พันคน')
    ax2.set_title('เปรียบเทียบรายไตรมาส: 2568 vs 2569', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.set_facecolor('#fafafa')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("#### 📋 ตารางพยากรณ์รายไตรมาส ปี 2569")
    pred_display = pred_2569[['Quarter_label', 'Predicted_vis']].copy()
    pred_display.columns = ['ไตรมาส', 'จำนวนนักท่องเที่ยว (พยากรณ์)']
    pred_display['จำนวนนักท่องเที่ยว (พยากรณ์)'] = pred_display['จำนวนนักท่องเที่ยว (พยากรณ์)'].map('{:,.0f}'.format)
    st.dataframe(pred_display, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 : EVENT IMPACT
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-header">🎯 Event Impact — ผลกระทบของ Event ต่อนักท่องเที่ยว</div>', unsafe_allow_html=True)

    # ── Event impact bar chart ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#f8f9fa')
    axes = axes.flatten()

    for idx, ec in enumerate(EVENT_COLS):
        ax = axes[idx]
        with_ev = df_quarterly[df_quarterly[ec] == 1]['Total_vis'].mean()
        without_ev = df_quarterly[df_quarterly[ec] == 0]['Total_vis'].mean()
        vals = [without_ev / 1e3, with_ev / 1e3]
        bars = ax.bar(['ไม่มี Event', f'มี {EVENT_LABELS[ec]}'],
                       vals, color=['#adb5bd', '#f5576c'], edgecolor='white')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{v:,.0f}K', ha='center', va='bottom', fontsize=10, fontweight='bold')
        diff_pct = (with_ev - without_ev) / without_ev * 100 if without_ev > 0 else 0
        ax.set_title(f'{EVENT_LABELS[ec]}\n(ผลต่าง: {diff_pct:+.1f}%)', fontsize=11, fontweight='bold')
        ax.set_ylabel('พันคน (เฉลี่ยต่อไตรมาส)')
        ax.set_facecolor('#fafafa')

    plt.suptitle('ผลกระทบของ Event ต่อนักท่องเที่ยวเฉลี่ยต่อไตรมาส', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── MotoGP year-over-year timeline ──
    st.markdown("#### 🏍️ MotoGP — จำนวนนักท่องเที่ยวในปีที่มี MotoGP vs ปีถัดไป")
    motogp_years = df_annual[df_annual['MotoGP'] == 1]['Year'].values

    if len(motogp_years) > 0:
        rows_mgp = []
        for yr in motogp_years:
            curr = df_annual[df_annual['Year'] == yr]['Total_vis'].values
            nxt = df_annual[df_annual['Year'] == yr + 1]['Total_vis'].values
            rows_mgp.append({
                'ปีที่มี MotoGP': yr,
                'นักท่องเที่ยวปีนั้น': int(curr[0]) if len(curr) > 0 else None,
                'นักท่องเที่ยวปีถัดไป': int(nxt[0]) if len(nxt) > 0 else None,
            })
        df_mgp = pd.DataFrame(rows_mgp).dropna()

        if len(df_mgp) > 0:
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            fig3.patch.set_facecolor('#f8f9fa')
            x = np.arange(len(df_mgp))
            w = 0.35
            ax3.bar(x - w/2, df_mgp['นักท่องเที่ยวปีนั้น'] / 1e6, w, label='ปีที่มี MotoGP', color='#f5576c')
            ax3.bar(x + w/2, df_mgp['นักท่องเที่ยวปีถัดไป'] / 1e6, w, label='ปีถัดไป', color='#667eea')
            ax3.set_xticks(x)
            ax3.set_xticklabels([f"ปี {r['ปีที่มี MotoGP']}" for _, r in df_mgp.iterrows()], rotation=30)
            ax3.set_ylabel('ล้านคน')
            ax3.set_title('MotoGP: นักท่องเที่ยวปีที่มี Event vs ปีถัดไป', fontsize=12, fontweight='bold')
            ax3.legend()
            ax3.set_facecolor('#fafafa')
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()

            st.dataframe(
                df_mgp.style.format({
                    'นักท่องเที่ยวปีนั้น': '{:,}',
                    'นักท่องเที่ยวปีถัดไป': '{:,}',
                }),
                use_container_width=True, hide_index=True
            )

    # ── Quarterly event impact over time ──
    st.markdown("#### 📈 แนวโน้มรายไตรมาสพร้อม Event overlay")
    fig4, ax4 = plt.subplots(figsize=(14, 5))
    fig4.patch.set_facecolor('#f8f9fa')

    df_q_plot = df_quarterly.sort_values(['Year', 'Quarter']).reset_index(drop=True)
    df_q_plot['period'] = df_q_plot['Year'].astype(str) + '-Q' + df_q_plot['Quarter'].astype(str)

    ax4.plot(range(len(df_q_plot)), df_q_plot['Total_vis'] / 1e5, 'o-',
             color='#667eea', linewidth=1.5, markersize=4, label='Total Visitors (แสนคน)')

    ev_colors2 = {'MotoGP': '#e74c3c', 'Covid': '#2c3e50', 'Marathon': '#27ae60', 'PhanomRung_Festival': '#f39c12'}
    for ec in EVENT_COLS:
        ev_idx = df_q_plot[df_q_plot[ec] == 1].index
        for i in ev_idx:
            ax4.axvline(i, color=ev_colors2[ec], linestyle='--', alpha=0.4, linewidth=1)

    # Legend patches
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=c, label=EVENT_LABELS[k]) for k, c in ev_colors2.items()]
    patches.insert(0, plt.Line2D([0], [0], color='#667eea', marker='o', label='Visitors'))
    ax4.legend(handles=patches, loc='upper left', fontsize=8)

    tick_step = max(1, len(df_q_plot) // 20)
    ax4.set_xticks(range(0, len(df_q_plot), tick_step))
    ax4.set_xticklabels(df_q_plot['period'].values[::tick_step], rotation=45, ha='right', fontsize=8)
    ax4.set_ylabel('แสนคน')
    ax4.set_title('แนวโน้มนักท่องเที่ยวรายไตรมาส พร้อม Event Markers', fontsize=12, fontweight='bold')
    ax4.set_facecolor('#fafafa')
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center;color:#888;font-size:0.85rem;">
    🏟️ Buriram Tourism Forecasting Dashboard | CRISP-DM Framework | 
    Models: Linear Regression, Ridge, Random Forest, Gradient Boosting |
    Metrics: MAE, RMSE, R²
</div>
""", unsafe_allow_html=True)
