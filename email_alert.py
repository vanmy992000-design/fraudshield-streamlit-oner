"""
FraudShield — Module xuất báo cáo HTML (thay thế gửi email)
=============================================================
Gọi: send_alert(df_result, filename, excel_bytes, log_fn)
Kết quả: Lưu file HTML vào thư mục output/reports/
"""

import os
from datetime import datetime
import pandas as pd


def _build_html(df_high: pd.DataFrame, df_all: pd.DataFrame, filename: str) -> str:
    """Tạo nội dung báo cáo HTML đẹp với bảng giao dịch HIGH risk."""
    n_high  = len(df_high)
    n_med   = (df_all["Risk_Level"] == "MEDIUM").sum()
    n_low   = (df_all["Risk_Level"] == "LOW").sum()
    n_total = len(df_all)
    now     = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    show_cols = [c for c in [
        "Transaction_ID", "Transaction_Amount", "Transaction_Type",
        "Merchant_Category", "Card_Type", "Is_International",
        "Unusual_Time", "Fraud_Probability", "Risk_Level",
    ] if c in df_high.columns]

    # Build table rows
    rows_html = ""
    for _, row in df_high[show_cols].iterrows():
        prob = row.get("Fraud_Probability", 0)
        cells = ""
        for col in show_cols:
            val = row[col]
            if col == "Fraud_Probability":
                cells += f"""
                <td>
                  <div class="prob-bar-wrap">
                    <div class="prob-bar" style="width:{prob*100:.0f}%"></div>
                  </div>
                  <span class="prob-text">{prob*100:.1f}%</span>
                </td>"""
            elif col == "Risk_Level":
                cells += f"<td><span class='badge-high'>🔴 HIGH</span></td>"
            elif col == "Transaction_Amount":
                cells += f"<td class='mono'>${val:,.2f}M</td>"
            else:
                cells += f"<td>{val}</td>"
        rows_html += f"<tr>{cells}</tr>"

    headers_html = "".join(
        f"<th>{c.replace('_', ' ')}</th>" for c in show_cols
    )

    # Risk distribution bar
    pct_high = n_high / n_total * 100 if n_total else 0
    pct_med  = n_med  / n_total * 100 if n_total else 0
    pct_low  = n_low  / n_total * 100 if n_total else 0

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FraudShield Report — {filename}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Segoe UI', Arial, sans-serif;
    background: #f0f4f8;
    color: #1e293b;
  }}

  /* ── Header ── */
  .header {{
    background: linear-gradient(135deg, #0f1117 0%, #1e293b 100%);
    padding: 28px 40px;
    display: flex;
    align-items: center;
    gap: 16px;
  }}
  .header-logo {{ font-size: 28px; font-weight: 800; color: #4fc3f7; letter-spacing: -0.5px; }}
  .header-sub  {{ color: #94a3b8; font-size: 13px; margin-top: 4px; }}

  /* ── Alert banner ── */
  .alert-banner {{
    background: #fee2e2;
    border-left: 5px solid #dc2626;
    padding: 18px 40px;
  }}
  .alert-title {{ color: #b91c1c; font-size: 18px; font-weight: 700; }}
  .alert-sub   {{ color: #7f1d1d; font-size: 13px; margin-top: 4px; }}

  /* ── Content ── */
  .content {{ padding: 32px 40px; max-width: 1400px; margin: 0 auto; }}

  /* ── KPI cards ── */
  .kpi-row {{ display: flex; gap: 16px; margin-bottom: 28px; flex-wrap: wrap; }}
  .kpi-card {{
    background: white;
    border-radius: 12px;
    padding: 20px 28px;
    flex: 1;
    min-width: 140px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    text-align: center;
  }}
  .kpi-num  {{ font-size: 32px; font-weight: 800; }}
  .kpi-label {{ font-size: 11px; color: #64748b; font-weight: 600;
                text-transform: uppercase; letter-spacing: 0.08em; margin-top: 6px; }}
  .red    {{ color: #dc2626; }}
  .yellow {{ color: #d97706; }}
  .green  {{ color: #16a34a; }}
  .dark   {{ color: #0f172a; }}

  /* ── Distribution bar ── */
  .dist-card {{
    background: white; border-radius: 12px; padding: 20px 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08); margin-bottom: 28px;
  }}
  .dist-title {{ font-weight: 700; font-size: 14px; margin-bottom: 14px; }}
  .dist-bar-wrap {{
    display: flex; height: 24px; border-radius: 8px; overflow: hidden; gap: 2px;
  }}
  .dist-seg-high   {{ background: #dc2626; height: 100%; border-radius: 6px 0 0 6px; }}
  .dist-seg-medium {{ background: #f59e0b; height: 100%; }}
  .dist-seg-low    {{ background: #22c55e; height: 100%; border-radius: 0 6px 6px 0; }}
  .dist-legend {{ display: flex; gap: 20px; margin-top: 10px; font-size: 12px; color: #64748b; }}
  .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 5px; vertical-align: middle; }}

  /* ── Table ── */
  .table-card {{
    background: white; border-radius: 12px; overflow: hidden;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08); margin-bottom: 24px;
  }}
  .table-title {{
    padding: 16px 20px; border-bottom: 1px solid #f1f5f9;
    font-weight: 700; font-size: 14px;
  }}
  .table-wrap {{ overflow-x: auto; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{
    padding: 10px 14px; background: #1e293b; color: white;
    text-align: left; font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.05em; white-space: nowrap;
  }}
  td {{ padding: 9px 14px; border-bottom: 1px solid #f1f5f9; }}
  tr:hover td {{ background: #f8fafc; }}
  tr:last-child td {{ border-bottom: none; }}
  .mono {{ font-family: monospace; }}

  .badge-high {{
    background: #fee2e2; color: #b91c1c;
    padding: 3px 10px; border-radius: 20px;
    font-weight: 700; font-size: 11px;
  }}
  .prob-bar-wrap {{
    display: inline-block; width: 70px; height: 7px;
    background: #fee2e2; border-radius: 4px; overflow: hidden;
    vertical-align: middle; margin-right: 6px;
  }}
  .prob-bar {{ height: 100%; background: #dc2626; border-radius: 4px; }}
  .prob-text {{ color: #dc2626; font-weight: 700; vertical-align: middle; }}

  /* ── Footer ── */
  .footer {{
    background: #e2e8f0; padding: 20px 40px; text-align: center;
    color: #94a3b8; font-size: 12px; margin-top: 40px;
  }}
  .generated-tag {{
    display: inline-block;
    background: #dbeafe; color: #1d4ed8;
    font-size: 11px; font-weight: 600;
    padding: 4px 12px; border-radius: 20px; margin-bottom: 16px;
  }}
</style>
</head>
<body>

<!-- Header -->
<div class="header">
  <div>
    <div class="header-logo">🛡️ FraudShield</div>
    <div class="header-sub">Banking Fraud Detection System · Automated Report</div>
  </div>
</div>

<!-- Alert banner -->
<div class="alert-banner">
  <div class="alert-title">⚠️ Phát hiện {n_high} giao dịch nghi ngờ gian lận</div>
  <div class="alert-sub">File: <strong>{filename}</strong> · Thời điểm: {now}</div>
</div>

<div class="content">

  <!-- KPI -->
  <div class="kpi-row">
    <div class="kpi-card">
      <div class="kpi-num red">{n_high}</div>
      <div class="kpi-label">🔴 HIGH Risk</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-num yellow">{n_med}</div>
      <div class="kpi-label">🟡 MEDIUM Risk</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-num green">{n_low}</div>
      <div class="kpi-label">🟢 LOW Risk</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-num dark">{n_total:,}</div>
      <div class="kpi-label">Tổng giao dịch</div>
    </div>
  </div>

  <!-- Distribution bar -->
  <div class="dist-card">
    <div class="dist-title">Phân bổ mức độ rủi ro</div>
    <div class="dist-bar-wrap">
      <div class="dist-seg-high"   style="width:{pct_high:.1f}%"></div>
      <div class="dist-seg-medium" style="width:{pct_med:.1f}%"></div>
      <div class="dist-seg-low"    style="width:{pct_low:.1f}%"></div>
    </div>
    <div class="dist-legend">
      <span><span class="dot" style="background:#dc2626"></span>HIGH {pct_high:.1f}%</span>
      <span><span class="dot" style="background:#f59e0b"></span>MEDIUM {pct_med:.1f}%</span>
      <span><span class="dot" style="background:#22c55e"></span>LOW {pct_low:.1f}%</span>
    </div>
  </div>

  <!-- Table -->
  <div class="table-card">
    <div class="table-title">🔴 Danh sách giao dịch HIGH Risk — cần kiểm tra ngay ({n_high} giao dịch)</div>
    <div class="table-wrap">
      <table>
        <thead><tr>{headers_html}</tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
  </div>

  <div class="generated-tag">📄 Báo cáo tự động · FraudShield · {now}</div>

</div>

<!-- Footer -->
<div class="footer">
  FraudShield Automated Report System · FAA4023 Final Assignment 2026<br>
  File được tạo tự động bởi scheduler. Không cần kết nối internet.
</div>

</body>
</html>
"""
    return html


def send_alert(df_result: pd.DataFrame,
               filename: str,
               excel_bytes: bytes,
               log_fn=print) -> bool:
    """
    Xuất báo cáo HTML thay thế gửi email.
    Trả về True nếu xuất thành công.
    """
    try:
        from config import MIN_HIGH_TO_ALERT
    except ImportError:
        MIN_HIGH_TO_ALERT = 1

    df_high = df_result[df_result["Risk_Level"] == "HIGH"].copy()
    df_high = df_high.sort_values("Fraud_Probability", ascending=False)
    n_high  = len(df_high)

    if n_high < MIN_HIGH_TO_ALERT:
        log_fn(f"[~] Chỉ có {n_high} HIGH risk (< ngưỡng {MIN_HIGH_TO_ALERT}) — bỏ qua xuất HTML")
        return False

    # Tạo thư mục output/reports/ nếu chưa có
    reports_dir = os.path.join("output", "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # Tên file theo timestamp
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem      = os.path.splitext(filename)[0]
    html_name = f"fraud_report_{stem}_{ts}.html"
    html_path = os.path.join(reports_dir, html_name)

    try:
        html_content = _build_html(df_high, df_result, filename)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        log_fn(f"[✓] Xuất báo cáo HTML → output/reports/{html_name}  ({n_high} HIGH risk)")
        return True
    except Exception as e:
        log_fn(f"[✗] Lỗi xuất HTML: {e}")
        return False