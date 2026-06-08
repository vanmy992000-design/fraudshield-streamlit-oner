"""
FraudShield — Bước 2: Feature Engineering (v2 — Cải tiến)
==========================================================
Các thay đổi so với v1:
  1. Thêm 8 features mới mạnh hơn
  2. Log-transform các cột có phân phối lệch (skewed)
  3. Interaction features giữa các yếu tố rủi ro

Input : data/cleaned_data.csv
Output: data/featured_data.csv
"""

import pandas as pd
import numpy as np
import os
import sys

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH  = os.path.join(BASE_DIR, "data", "cleaned_data.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "featured_data.csv")


def log(msg):
    print(f"  {msg}")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ══════════════════════════════════════════════════════════
    # FEATURES GỐC (giữ nguyên)
    # ══════════════════════════════════════════════════════════

    # Feature 1: is_high_risk_combo
    df["is_high_risk_combo"] = (
        (df["Is_International"] == "Yes") & (df["Unusual_Time"] == "Yes")
    ).astype(int)
    log("[✓] Feature 1 : is_high_risk_combo")

    # Feature 2: amount_vs_avg_ratio
    df["amount_vs_avg_ratio"] = np.where(
        df["Avg_Transaction_Amount"] > 0,
        df["Transaction_Amount"] / df["Avg_Transaction_Amount"],
        1.0,
    )
    log("[✓] Feature 2 : amount_vs_avg_ratio")

    # Feature 3: balance_vs_amount_ratio
    df["balance_vs_amount_ratio"] = np.where(
        df["Transaction_Amount"] > 0,
        df["Account_Balance"] / df["Transaction_Amount"],
        df["Account_Balance"],
    )
    log("[✓] Feature 3 : balance_vs_amount_ratio")

    # Feature 4: is_night
    df["is_night"] = ((df["Hour"] < 6) | (df["Hour"] > 22)).astype(int)
    log("[✓] Feature 4 : is_night")

    # Feature 5: transaction_velocity
    df["transaction_velocity"] = np.where(
        df["Weekly_Transaction_Count"] > 0,
        df["Daily_Transaction_Count"] / df["Weekly_Transaction_Count"] * 7,
        1.0,
    )
    log("[✓] Feature 5 : transaction_velocity")

    # Feature 6: is_large_transaction
    df["is_large_transaction"] = (
        df["Transaction_Amount"] > df["Max_Transaction_Last_24h"]
    ).astype(int)
    log("[✓] Feature 6 : is_large_transaction")

    # ══════════════════════════════════════════════════════════
    # FEATURES MỚI (v2)
    # ══════════════════════════════════════════════════════════

    # Feature 7: risk_score_sum — cộng dồn các tín hiệu rủi ro
    # Càng nhiều tín hiệu cùng lúc → khả năng fraud càng cao
    df["risk_score_sum"] = (
        df["is_high_risk_combo"]
        + df["is_night"]
        + df["is_large_transaction"]
        + (df["Is_International"] == "Yes").astype(int)
        + (df["Unusual_Time"] == "Yes").astype(int)
        + (df.get("Is_New_Merchant", "No") == "Yes").astype(int)
    )
    log("[✓] Feature 7 : risk_score_sum (tổng tín hiệu rủi ro)")

    # Feature 8: amount_vs_max_ratio — so sánh với max 24h
    df["amount_vs_max_ratio"] = np.where(
        df["Max_Transaction_Last_24h"] > 0,
        df["Transaction_Amount"] / df["Max_Transaction_Last_24h"],
        1.0,
    )
    log("[✓] Feature 8 : amount_vs_max_ratio = Amount / Max_Last_24h")

    # Feature 9: has_previous_fraud — từng có fraud trước đó
    if "Previous_Fraud_Count" in df.columns:
        df["has_previous_fraud"] = (df["Previous_Fraud_Count"] > 0).astype(int)
        log("[✓] Feature 9 : has_previous_fraud")
    else:
        df["has_previous_fraud"] = 0
        log("[!] Feature 9 : has_previous_fraud = 0 (thiếu cột)")

    # Feature 10: failed_ratio — tỷ lệ giao dịch thất bại / tổng ngày
    if "Failed_Transaction_Count" in df.columns and "Daily_Transaction_Count" in df.columns:
        df["failed_ratio"] = np.where(
            df["Daily_Transaction_Count"] > 0,
            df["Failed_Transaction_Count"] / df["Daily_Transaction_Count"],
            0.0,
        )
    else:
        df["failed_ratio"] = 0.0
    log("[✓] Feature 10: failed_ratio = Failed / Daily_Count")

    # Feature 11: log_amount — log-transform để giảm skewness
    df["log_amount"] = np.log1p(df["Transaction_Amount"])
    log("[✓] Feature 11: log_amount = log1p(Transaction_Amount)")

    # Feature 12: log_balance
    df["log_balance"] = np.log1p(df["Account_Balance"].clip(lower=0))
    log("[✓] Feature 12: log_balance = log1p(Account_Balance)")

    # Feature 13: intl_x_new_merchant — giao dịch quốc tế tại merchant mới
    if "Is_New_Merchant" in df.columns:
        df["intl_x_new_merchant"] = (
            (df["Is_International"] == "Yes") & (df["Is_New_Merchant"] == "Yes")
        ).astype(int)
    else:
        df["intl_x_new_merchant"] = 0
    log("[✓] Feature 13: intl_x_new_merchant (International AND New_Merchant)")

    # Feature 14: high_amount_night — số tiền lớn vào ban đêm
    df["high_amount_night"] = (
        df["is_night"] * df["amount_vs_avg_ratio"]
    )
    log("[✓] Feature 14: high_amount_night = is_night × amount_vs_avg_ratio")

    # ══════════════════════════════════════════════════════════
    # ENCODE
    # ══════════════════════════════════════════════════════════

    df["Is_International_bin"] = (df["Is_International"] == "Yes").astype(int)
    df["Is_New_Merchant_bin"]  = (df.get("Is_New_Merchant", pd.Series(["No"]*len(df))) == "Yes").astype(int)
    df["Unusual_Time_bin"]     = (df["Unusual_Time"] == "Yes").astype(int)
    df["Card_Type_bin"]        = (df["Card_Type"] == "Credit").astype(int)
    log("[✓] Encode binary: Is_International, Is_New_Merchant, Unusual_Time, Card_Type")

    df = pd.get_dummies(df, columns=["Transaction_Type"],   prefix="TxType",   drop_first=False)
    log("[✓] One-hot: Transaction_Type → TxType_*")

    df = pd.get_dummies(df, columns=["Merchant_Category"],  prefix="Merchant",  drop_first=False)
    log("[✓] One-hot: Merchant_Category → Merchant_*")

    if "Fraud_Label" in df.columns:
        df["Fraud_Binary"] = (df["Fraud_Label"] == "Fraud").astype(int)
        log("[✓] Fraud_Label → Fraud_Binary")

    return df


def run(input_path=INPUT_PATH, output_path=OUTPUT_PATH):
    print("\n" + "═" * 60)
    print("  FraudShield │ 02_feature_engineering.py  (v2)")
    print("═" * 60)

    df = pd.read_csv(input_path)
    log(f"[✓] Đọc {len(df):,} dòng từ {os.path.basename(input_path)}")

    df_feat = engineer_features(df)
    df_feat.to_csv(output_path, index=False)

    n_new = df_feat.shape[1] - df.shape[1]
    log(f"[✓] Tạo thêm {n_new} features mới → tổng {df_feat.shape[1]} cột")
    log(f"[✓] Xuất → {os.path.basename(output_path)}")
    print("═" * 60 + "\n")

    return df_feat


if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else INPUT_PATH
    out = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_PATH
    run(inp, out)