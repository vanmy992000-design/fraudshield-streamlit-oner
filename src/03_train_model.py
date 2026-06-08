"""
FraudShield — Bước 3: Huấn luyện mô hình XGBoost (v3)
=======================================================
Thay Random Forest → XGBoost:
  - scale_pos_weight tự động xử lý imbalanced data
  - Gradient boosting bắt pattern tốt hơn RF trên tabular data
  - Kỳ vọng AUC tăng từ ~0.57 → ~0.80+

Cài trước khi chạy:
  pip install xgboost imbalanced-learn

Input : data/featured_data.csv
Output: model/fraud_model.pkl + model/model_meta.json
"""

import pandas as pd
import numpy as np
import os, sys, joblib, json
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score, precision_score, recall_score,
    precision_recall_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "data",   "featured_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model",  "fraud_model.pkl")
META_PATH  = os.path.join(BASE_DIR, "model",  "model_meta.json")
EVAL_PATH  = os.path.join(BASE_DIR, "output", "model_evaluation.txt")
PLOT_DIR   = os.path.join(BASE_DIR, "output")

FEATURE_COLS = [
    "Transaction_Amount", "Account_Balance", "Distance_From_Home",
    "Daily_Transaction_Count", "Weekly_Transaction_Count",
    "Avg_Transaction_Amount", "Max_Transaction_Last_24h",
    "Failed_Transaction_Count", "Previous_Fraud_Count",
    "Hour", "Month", "DayOfWeek",
    "is_high_risk_combo", "amount_vs_avg_ratio", "balance_vs_amount_ratio",
    "is_night", "transaction_velocity", "is_large_transaction",
    "risk_score_sum", "amount_vs_max_ratio", "has_previous_fraud",
    "failed_ratio", "log_amount", "log_balance",
    "intl_x_new_merchant", "high_amount_night",
    "Is_International_bin", "Is_New_Merchant_bin",
    "Unusual_Time_bin", "Card_Type_bin",
]

def log(msg): print(f"  {msg}")

def get_feature_cols(df):
    base    = [c for c in FEATURE_COLS if c in df.columns]
    dummies = [c for c in df.columns if c.startswith(("TxType_", "Merchant_"))]
    return base + dummies

def find_best_threshold(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = np.where(
        (precisions + recalls) == 0, 0,
        2 * precisions * recalls / (precisions + recalls)
    )
    best_idx = np.argmax(f1_scores[:-1])
    return thresholds[best_idx], f1_scores[best_idx]


def train(input_path=INPUT_PATH, model_path=MODEL_PATH, eval_path=EVAL_PATH):
    print("\n" + "═" * 60)
    print("  FraudShield │ 03_train_model.py  (v3 — XGBoost)")
    print("═" * 60)

    # 1. Load data
    df = pd.read_csv(input_path)
    log(f"[✓] Đọc {len(df):,} dòng")

    feature_cols = get_feature_cols(df)
    X = df[feature_cols].fillna(0)
    y = df["Fraud_Binary"]

    n_normal = (y == 0).sum()
    n_fraud  = (y == 1).sum()
    scale_pos_weight = round(n_normal / n_fraud, 2)
    log(f"[✓] {len(feature_cols)} features | Normal={n_normal:,} | Fraud={n_fraud:,} | scale_pos_weight={scale_pos_weight}")

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log(f"[✓] Train={len(X_train):,} | Test={len(X_test):,}")

    # 3. SMOTE (nếu có)
    try:
        from imblearn.over_sampling import SMOTE
        X_train, y_train = SMOTE(random_state=42, k_neighbors=5).fit_resample(X_train, y_train)
        log(f"[✓] SMOTE: {len(X_train):,} samples sau oversample")
    except ImportError:
        log("[!] imbalanced-learn chưa cài → dùng scale_pos_weight thay thế")
        log("    pip install imbalanced-learn")

    # 4. Train XGBoost (fallback RandomForest nếu chưa cài)
    try:
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            gamma=1,
            reg_alpha=0.1,
            reg_lambda=1,
            scale_pos_weight=scale_pos_weight,
            eval_metric="auc",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        model_name = "XGBoost"
        log("[✓] Dùng XGBoost (n=500, lr=0.05, scale_pos_weight tự động)")
    except ImportError:
        log("[!] XGBoost chưa cài → fallback RandomForest")
        log("    Để đạt kết quả tốt nhất: pip install xgboost")
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=5,
            max_features="sqrt", class_weight="balanced",
            random_state=42, n_jobs=-1,
        )
        model_name = "RandomForest (fallback)"

    log(f"[…] Đang train {model_name}…")
    model.fit(X_train, y_train)
    log("[✓] Hoàn thành")

    # 5. Đánh giá
    y_prob = model.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)

    # threshold 0.5
    y05    = (y_prob >= 0.5).astype(int)
    log(f"[threshold=0.50] AUC={auc:.4f}  F1={f1_score(y_test,y05):.4f}  "
        f"Recall={recall_score(y_test,y05):.4f}  Prec={precision_score(y_test,y05,zero_division=0):.4f}")

    # threshold tối ưu
    best_thresh, _ = find_best_threshold(y_test, y_prob)
    y_pred  = (y_prob >= best_thresh).astype(int)
    f1      = f1_score(y_test, y_pred)
    recall  = recall_score(y_test, y_pred)
    prec    = precision_score(y_test, y_pred, zero_division=0)
    cm      = confusion_matrix(y_test, y_pred)
    report  = classification_report(y_test, y_pred, target_names=["Normal","Fraud"])
    log(f"[threshold={best_thresh:.2f}] AUC={auc:.4f}  F1={f1:.4f}  "
        f"Recall={recall:.4f}  Prec={prec:.4f}  ← DÙNG THRESHOLD NÀY")

    # Cross-validation
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    log(f"[✓] CV AUC 5-fold: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

    # 6. Vẽ biểu đồ
    BG = "#0D1B2A"; BG2 = "#112240"; ACCENT = "#00D4FF"; MUTED = "#5B8DB8"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(BG)

    # Confusion matrix
    ax = axes[0]; ax.set_facecolor(BG2)
    ax.imshow(cm, cmap="Blues", interpolation="nearest")
    ax.set_title("Confusion Matrix", color="white", fontsize=14, pad=12, fontweight="600")
    ax.set_xlabel("Predicted", color=MUTED); ax.set_ylabel("Actual", color=MUTED)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Normal","Fraud"], color="white")
    ax.set_yticklabels(["Normal","Fraud"], color="white")
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#1E3A5F")
    thresh_cm = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    color="white" if cm[i,j] > thresh_cm else BG,
                    fontsize=18, fontweight="bold")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax2 = axes[1]; ax2.set_facecolor(BG2)
    ax2.plot(fpr, tpr, color=ACCENT, lw=2.5, label=f"AUC = {auc:.3f}")
    ax2.plot([0,1],[0,1],"--", color="#1E3A5F", lw=1.5)
    ax2.fill_between(fpr, tpr, alpha=0.12, color=ACCENT)
    ax2.set_xlabel("False Positive Rate", color=MUTED)
    ax2.set_ylabel("True Positive Rate", color=MUTED)
    ax2.set_title("ROC Curve", color="white", fontsize=14, pad=12, fontweight="600")
    ax2.tick_params(colors=MUTED)
    ax2.legend(facecolor=BG2, labelcolor="white", fontsize=11)
    for sp in ax2.spines.values(): sp.set_edgecolor("#1E3A5F")
    ax2.grid(color="#1E3A5F", linewidth=0.5, alpha=0.5)
    fig.tight_layout(pad=2)
    plt.savefig(os.path.join(PLOT_DIR, "confusion_matrix_roc.png"), dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    # Feature importance
    try:
        importances = pd.Series(model.feature_importances_, index=feature_cols)
        top10 = importances.nlargest(10)
        fig2, ax3 = plt.subplots(figsize=(10, 5))
        fig2.patch.set_facecolor(BG); ax3.set_facecolor(BG2)
        colors_fi = [ACCENT if i < 3 else "#0099CC" if i < 6 else "#006B8F" for i in range(10)]
        bars = ax3.barh(top10.index[::-1], top10.values[::-1],
                        color=colors_fi[::-1], edgecolor="none", height=0.6)
        for bar, val in zip(bars, top10.values[::-1]):
            ax3.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                     f"{val:.4f}", va="center", color=ACCENT,
                     fontsize=9, fontweight="600", fontfamily="monospace")
        ax3.set_xlabel("Importance Score", color=MUTED)
        ax3.set_title("Top 10 Feature Importance", color="white", fontsize=14, pad=12, fontweight="600")
        ax3.tick_params(colors=MUTED)
        for sp in ax3.spines.values(): sp.set_edgecolor("#1E3A5F")
        ax3.grid(axis="x", color="#1E3A5F", linewidth=0.5, alpha=0.5)
        fig2.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "feature_importance.png"), dpi=150,
                    bbox_inches="tight", facecolor=fig2.get_facecolor())
        plt.close()
    except Exception:
        pass

    log("[✓] Lưu biểu đồ → output/")

    # 7. Lưu model + meta
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    meta = {
        "feature_cols":    feature_cols,
        "best_threshold":  round(float(best_thresh), 4),
        "model_type":      model_name,
        "metrics": {
            "AUC":         round(auc,    4),
            "F1":          round(f1,     4),
            "Recall":      round(recall, 4),
            "Precision":   round(prec,   4),
            "CV_AUC_mean": round(float(cv_auc.mean()), 4),
            "CV_AUC_std":  round(float(cv_auc.std()),  4),
        },
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    log(f"[✓] Lưu model ({model_name}) → {os.path.basename(model_path)}")

    # 8. Báo cáo text
    with open(eval_path, "w", encoding="utf-8") as f:
        f.write(f"FraudShield — Model Evaluation ({model_name})\n")
        f.write(f"Trained at : {meta['trained_at']}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"AUC-ROC    : {auc:.4f}\n")
        f.write(f"Threshold  : {best_thresh:.4f}\n")
        f.write(f"F1 Score   : {f1:.4f}\n")
        f.write(f"Recall     : {recall:.4f}\n")
        f.write(f"Precision  : {prec:.4f}\n")
        f.write(f"CV AUC     : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}\n\n")
        f.write("Classification Report:\n" + report + "\n")
        f.write("Confusion Matrix:\n" + str(cm) + "\n")

    print("═" * 60 + "\n")
    return model, feature_cols, meta


if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else INPUT_PATH
    train(inp)