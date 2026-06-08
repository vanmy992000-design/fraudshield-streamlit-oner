# 🛡️ FraudShield — Banking Fraud Detection System
> FAA4023 Final Assignment · Semester 2 · AY 2025–2026

---

## 📁 Cấu trúc thư mục

```
FraudShield/
├── data/
│   └── FraudShield_Banking_Data.csv   ← Dataset gốc
├── src/
│   ├── 01_data_cleaning.py            ← Làm sạch dữ liệu tự động
│   ├── 02_feature_engineering.py      ← Tạo 6 features mới
│   ├── 03_train_model.py              ← Huấn luyện Random Forest
│   └── 04_predict.py                  ← Dự đoán file mới → Excel
├── model/
│   ├── fraud_model.pkl                ← Model đã train (auto-generated)
│   └── model_meta.json               ← Metrics + feature list
├── output/
│   ├── fraud_report.xlsx              ← Báo cáo Excel (auto-generated)
│   ├── confusion_matrix_roc.png       ← Biểu đồ đánh giá model
│   └── feature_importance.png        ← Top 10 features
├── app.py                             ← Dashboard Streamlit
├── requirements.txt
└── README.md
```

---

## 🚀 Hướng dẫn cài đặt & chạy

### 1. Cài môi trường

```bash
# Tạo virtual environment (khuyến nghị)
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# Cài dependencies
pip install -r requirements.txt
```

### 2. Chạy Pipeline (cách 1 — từng bước)

```bash
# Bước 1: Làm sạch dữ liệu
python src/01_data_cleaning.py

# Bước 2: Tạo features
python src/02_feature_engineering.py

# Bước 3: Huấn luyện model
python src/03_train_model.py

# Bước 4: Dự đoán file mới (thay tên file tùy ý)
python src/04_predict.py data/FraudShield_Banking_Data.csv output/fraud_report.xlsx
```

### 3. Chạy Dashboard Streamlit (cách 2 — giao diện trực quan)

```bash
streamlit run app.py
```

Trình duyệt sẽ tự mở tại `http://localhost:8501`

> **Lưu ý:** Trong dashboard, nhấn **"▶ Chạy Pipeline Đầy đủ"** ở sidebar để train model trước,
> sau đó upload CSV ở Tab 3 để dự đoán.

---

## 📊 Features mới tạo ra

| Feature | Công thức | Ý nghĩa |
|---|---|---|
| `is_high_risk_combo` | `Is_International=Yes AND Unusual_Time=Yes` | Tổ hợp rủi ro cao nhất (fraud rate 7.3%) |
| `amount_vs_avg_ratio` | `Transaction_Amount / Avg_Transaction_Amount` | Giao dịch đột biến so với lịch sử |
| `balance_vs_amount_ratio` | `Account_Balance / Transaction_Amount` | Tỷ lệ số dư / giao dịch |
| `is_night` | `Hour < 6 OR Hour > 22` | Giao dịch đêm khuya |
| `transaction_velocity` | `Daily_Count / Weekly_Count * 7` | Tốc độ giao dịch bất thường |
| `is_large_transaction` | `Amount > Max_Last_24h` | Giao dịch lớn nhất trong 24h |

---

## 🤖 Mô hình

- **Algorithm:** Random Forest (n_estimators=100, max_depth=10)
- **Imbalanced handling:** `class_weight='balanced'`
- **Chỉ tiêu chính:** Recall của class Fraud (bỏ sót Fraud nguy hiểm hơn cảnh báo nhầm)
- **Kỳ vọng:** AUC > 0.75, Recall(Fraud) > 0.65

---

## 🏷️ Risk Level

| Score | Level | Ý nghĩa |
|---|---|---|
| < 30% | 🟢 LOW | Giao dịch bình thường |
| 30–60% | 🟡 MEDIUM | Cần theo dõi |
| > 60% | 🔴 HIGH | Nghi ngờ fraud — cần kiểm tra ngay |

---

## ⚠️ AI Disclosure

Một số phần của assignment này có sử dụng AI (Claude by Anthropic) để:
- Hỗ trợ viết code Python/Streamlit
- Gợi ý cấu trúc pipeline và feature engineering
- Review và debug code

Toàn bộ phân tích, nhận xét kết quả, và nội dung báo cáo là do sinh viên tự thực hiện.
