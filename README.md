# 🏏 Buriram Tourism Forecasting Dashboard

ระบบวิเคราะห์และพยากรณ์นักท่องเที่ยวจังหวัดบุรีรัมย์  
แนวคิด **CRISP-DM 6 ขั้นตอน** | Python + Streamlit

---

## 📋 โครงสร้างโปรเจกต์

```
buriram_tourism_dashboard/
├── app.py                          # Streamlit main app
├── dataCI02-09-03-2569.csv         # ไฟล์ข้อมูล (วางไว้ที่นี่)
├── requirements.txt
├── .streamlit/
│   └── config.toml                 # Theme settings
└── README.md
```

---

## 🚀 วิธี Deploy บน Streamlit Cloud ผ่าน GitHub

### ขั้นตอนที่ 1: เตรียม GitHub Repository
1. สร้าง repo ใหม่ที่ GitHub เช่น `buriram-tourism-dashboard`
2. Upload ไฟล์ทั้งหมดในโฟลเดอร์นี้ขึ้น repo
3. **ต้องมีไฟล์ `dataCI02-09-03-2569.csv` อยู่ใน root ของ repo ด้วย**

### ขั้นตอนที่ 2: Deploy บน Streamlit Cloud
1. ไปที่ [https://share.streamlit.io](https://share.streamlit.io)
2. เชื่อม GitHub account
3. กด **"New app"**
4. เลือก repo → branch `main` → Main file: `app.py`
5. กด **"Deploy"**

---

## 💻 วิธีรันบนเครื่อง (Local)

### VS Code / Terminal
```bash
# 1. Clone หรือ download โฟลเดอร์มาไว้ในเครื่อง
cd buriram_tourism_dashboard

# 2. ติดตั้ง dependencies
pip install -r requirements.txt

# 3. รัน app
streamlit run app.py
```

### Google Colab
```python
# ใน Colab cell:
!pip install streamlit pyngrok -q

# Upload ไฟล์ผ่าน Colab:
from google.colab import files
uploaded = files.upload()  # อัพโหลด dataCI02-09-03-2569.csv

# รัน streamlit ผ่าน pyngrok:
!pip install pyngrok
from pyngrok import ngrok
import subprocess, time

proc = subprocess.Popen(['streamlit', 'run', 'app.py',
                         '--server.port', '8501',
                         '--server.headless', 'true'])
time.sleep(3)
public_url = ngrok.connect(8501)
print(f"🌐 Dashboard URL: {public_url}")
```

---

## 📊 Features ของ Dashboard

| Tab | รายละเอียด |
|-----|-----------|
| 📋 CRISP-DM Overview | แสดง 6 ขั้นตอน + Data Cleaning notes |
| 🔍 Data Exploration | Annual trend, Quarterly heatmap, Thai/Foreign split |
| 📅 Monthly Analysis | กราฟรายเดือนแต่ละปี (2567-2568), เปรียบเทียบ 2 ปี |
| 🤖 Model & Evaluation | MAE, RMSE, R² ของทั้ง 4 โมเดล + Feature Importance |
| 🔮 Forecast 2569 | พยากรณ์รายไตรมาส + กราฟเปรียบเทียบกับปี 2568 |
| 🎯 Event Impact | ผลกระทบ MotoGP/COVID/Marathon/ผานรุ้ง ต่อนักท่องเที่ยว |

---

## 🤖 ML Models

- **Linear Regression** — พื้นฐาน, ตีความง่าย
- **Ridge Regression** — Linear + L2 regularization ป้องกัน overfitting
- **Random Forest Regression** — Ensemble tree-based, จัดการ non-linearity
- **Gradient Boosting Regression** — Boosting, มักได้ผลดีที่สุด

**Auto-Select** จะเลือกโมเดลที่ได้ R² สูงสุด

---

## 📐 Features ที่ใช้ใน Model

| Feature | คำอธิบาย |
|---------|---------|
| `vis_lag1` | นักท่องเที่ยวไตรมาสก่อนหน้า |
| `vis_lag2` | นักท่องเที่ยว 2 ไตรมาสก่อน |
| `vis_lag4` | นักท่องเที่ยวไตรมาสเดียวกันปีก่อน |
| `Q1-Q4` | Quarter dummy variables |
| `year_trend` | แนวโน้มตามเวลา |
| `MotoGP`, `Covid`, `Marathon`, `PhanomRung_Festival` | Event flags |

---

## 📁 ข้อมูล

- ที่มา: ข้อมูลนักท่องเที่ยวจังหวัดบุรีรัมย์ ปี พ.ศ. 2556–2568
- ความถี่: รายไตรมาส (ปี 2556–2568) + รายเดือน (ปี 2567–2568)
- Metrics: Total visitors, Thai/Foreign split, Occupancy rate, Revenue
