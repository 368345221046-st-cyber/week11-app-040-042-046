@echo off
chcp 65001 >nul
REM ── เปิดแดชบอร์ด AI ทำนายลูกค้าตอบรับเงินฝาก (ทีม OKB EiEi 040-042-046) ──
cd /d "%~dp0"
echo กำลังเปิดแอป... เบราว์เซอร์จะเปิดที่ http://localhost:8501
echo (ปิดหน้าต่างนี้เพื่อหยุดแอป)
python -m streamlit run app.py --server.port 8501 --browser.gatherUsageStats false
pause
