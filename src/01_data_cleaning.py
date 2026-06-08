"""
FraudShield — Bước 1: Làm sạch dữ liệu tự động
================================================
Input : data/FraudShield_Banking_Data.csv
Output: data/cleaned_data.csv
"""

import pandas as pd
import numpy as np
import os
import sys

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH  = os.path.join(BASE_DIR, "data", "FraudShield_Banking_Data.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "cleaned_data.csv")


def log(msg): print(f"  {msg}")


def clean_data(input_path=INPUT_PATH, output_path=OUTPUT_PATH):
    print("\n" + "═" * 60)
    print("  FraudShield │ 01_data_cleaning.py")
    print("═" * 60)

    df = pd.read_csv(input_path)
    n_raw = len(df)
    log(f"[✓] Đọc {n_raw:,} dòng từ: {os.path.basename(input_path)}")

    # Đổi tên cột
    rename_map = {
        "Transaction_Amount (in Million)": "Transaction_Amount",
        "Account_Balance (in Million)":    "Account_Balance",
        "Avg_Transaction_Amount (in Million)": "Avg_Transaction_Amount",
        "Max_Transaction_Last_24h (in Million)": "Max_Transaction_Last_24h",
        "Is_International_Transaction": "Is_International",
        "Unusual_Time_Transaction": "Unusual_Time",
    }
    df = df.rename(columns=rename_map)

    # Xóa dòng thiếu nhãn
    n_missing_label = df["Fraud_Label"].isna().sum()
    df = df.dropna(subset=["Fraud_Label"]).copy()
    log(f"[✓] Xóa {n_missing_label} dòng thiếu nhãn Fraud_Label")

    # Đếm missing trước khi impute
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.select_dtypes(include=["object", "string"]).columns
                if c != "Fraud_Label"]
    n_missing_before = df[num_cols + cat_cols].isna().sum().sum()

    # Impute số → median
    for col in num_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Impute phân loại → mode
    for col in cat_cols:
        if df[col].isna().any():
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])

    log(f"[✓] Impute {n_missing_before:,} giá trị thiếu (số→median, phân loại→mode)")

    # Parse date
    df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"], errors="coerce")
    df["Month"]     = df["Transaction_Date"].dt.month.fillna(1).astype(int)
    df["DayOfWeek"] = df["Transaction_Date"].dt.dayofweek.fillna(0).astype(int)
    df["DayName"]   = df["Transaction_Date"].dt.day_name().fillna("Monday")
    log("[✓] Tách Transaction_Date → Month, DayOfWeek, DayName")

    # Parse hour
    def parse_hour(t):
        try:
            return int(str(t).split(":")[0])
        except Exception:
            return np.nan

    hour_series = df["Transaction_Time"].apply(parse_hour)
    hour_median = hour_series.median()
    if pd.isna(hour_median):
        hour_median = 12
    df["Hour"] = hour_series.fillna(hour_median).astype(int)
    log("[✓] Tách Transaction_Time → Hour (0–23)")

    # Chuẩn hóa Yes/No
    for col in ["Is_International", "Is_New_Merchant", "Unusual_Time"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.capitalize()
            df[col] = df[col].map({"Yes": "Yes", "No": "No"}).fillna("No")

    df.to_csv(output_path, index=False)
    log(f"[✓] Xuất {len(df):,} dòng sạch → {os.path.basename(output_path)}")
    print("═" * 60 + "\n")
    return df


if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else INPUT_PATH
    out = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_PATH
    clean_data(inp, out)
