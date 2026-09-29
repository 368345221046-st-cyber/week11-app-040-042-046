# -*- coding: utf-8 -*-
"""Streamlit app — ทีม OKB EiEi (040-042-046) · AI จำแนกลูกค้าที่มีแนวโน้มตอบรับข้อเสนอเงินฝาก"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI ทำนายลูกค้าตอบรับเงินฝาก | ทีม OKB EiEi", page_icon="🏦", layout="wide")
BASE = Path(__file__).resolve().parent


@st.cache_resource
def load_models():
    return (joblib.load(BASE / "team_model.joblib"),
            joblib.load(BASE / "team_model_precall.joblib"))


@st.cache_data
def load_meta():
    with open(BASE / "team_features.json", encoding="utf-8") as f:
        return json.load(f)


model, model_pre = load_models()
meta = load_meta()
FEATURES = meta["features"]
FEATURES_PRE = meta.get("features_precall", [f for f in FEATURES if f != "duration"])
TEAM = [(c, n) for c, n in meta.get("members", [["040", "ศักรินทร์ เกิดศิริ"],
                                                ["042", "สมชาย เมืองแก้ว"],
                                                ["046", "อาทินันท์ วรรณารุณ"]])]
ACC = float(meta.get("test_accuracy", 0.8545))
F1 = float(meta.get("f1", 0.8519))
AUC = float(meta.get("roc_auc", 0.9219))
PREC = float(meta.get("precision", 0.8223))
REC = float(meta.get("recall", 0.8837))
CV3 = float(meta.get("cv3", 0.8025))
N_ROWS = int(meta.get("n_rows", 11162))
RATE = float(meta.get("deposit_rate", 0.4738))
IMP = meta.get("feature_importance", {})
IMP_PRE = meta.get("feature_importance_precall", {})
PRE = meta.get("precall", {"test_acc": 0.7313, "f1": 0.6812, "roc_auc": 0.7882})
DT = meta.get("dt_holdout", {"test_acc": 0.803, "f1": 0.794})
CATS = meta.get("categories", {})
NUMR = meta.get("numeric_range", {})
NICE = {"job": "อาชีพ (Job)", "marital": "สถานภาพสมรส", "education": "ระดับการศึกษา",
        "contact": "ช่องทางติดต่อ", "month": "เดือนที่ติดต่อครั้งล่าสุด",
        "poutcome": "ผลแคมเปญครั้งก่อน"}
NICE_NUM = {"age": "อายุ (ปี)", "balance": "ยอดเงินในบัญชี (ยูโร)", "day": "วันที่ติดต่อ (1-31)",
            "duration": "ความยาวสายสนทนา (วินาที)", "campaign": "จำนวนครั้งที่ติดต่อในแคมเปญนี้",
            "pdays": "จำนวนวันตั้งแต่ติดต่อครั้งก่อน (-1 = ไม่เคย)", "previous": "จำนวนครั้งที่ติดต่อก่อนหน้า"}

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'IBM Plex Sans Thai','Leelawadee UI',Tahoma,sans-serif; }
  .hero { background: linear-gradient(135deg,#052E22 0%,#065F46 50%,#10B981 100%);
          border-radius: 22px; padding: 28px 32px 24px; color:#fff; position: relative; overflow: hidden;
          box-shadow: 0 24px 48px -28px rgba(6,95,70,.75); margin-bottom: 18px; }
  .hero:after { content:""; position:absolute; right:-70px; top:-70px; width:250px; height:250px;
                background: radial-gradient(circle, rgba(255,255,255,.22), transparent 65%); border-radius:50%; }
  .hero .kicker { font-size:.82rem; letter-spacing:2.2px; text-transform:uppercase; opacity:.88; }
  .hero h1 { font-size:2rem; font-weight:700; margin:6px 0 8px; line-height:1.25; }
  .hero p { margin:0; opacity:.95; font-size:.97rem; line-height:1.6; }
  .chips { margin-top:14px; }
  .chip { display:inline-block; background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.32);
          padding:6px 14px; border-radius:999px; font-size:.84rem; margin:0 8px 8px 0; }
  .chip.solid { background:#FFFFFF; color:#065F46 !important; border-color:#FFFFFF; font-weight:600; }
  div[data-testid="stVerticalBlockBorderWrapper"] { border-radius:16px !important;
      border:1px solid #D1FAE5 !important; background:#F8FEFB !important; padding:6px 4px; }
  div.stFormSubmitButton > button, div.stButton > button {
      background: linear-gradient(135deg,#065F46,#10B981) !important; color:#fff !important; border:none !important;
      border-radius:12px !important; font-weight:600 !important; padding:.62rem 1rem !important; }
  [data-testid="stMetric"] { background: linear-gradient(180deg,#FFFFFF,#ECFDF5);
      border:1px solid #D1FAE5; border-radius:14px; padding:14px 18px; }
  [data-testid="stMetricValue"], [data-testid="stMetricValue"] * { color:#065F46 !important; }
  [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * { color:#4B6B60 !important; }
  [data-testid="stAppViewContainer"], .stApp { background:#F6FDFA !important; color-scheme: light; }
  [data-testid="stHeader"] { background: transparent !important; }
  .stApp { color:#122A22; }
  [data-testid="stMarkdownContainer"] > p, [data-testid="stMarkdownContainer"] li,
  [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] * { color:#122A22; }
  .stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stCaptionContainer"] *,
  .stApp small { color:#5C7A70 !important; }
  .hero h1, .hero p, .hero .kicker, .hero .chip { color:#FFFFFF !important; }
  .hero .chip.solid { color:#065F46 !important; }
  .badge { border-radius:16px; padding:18px 20px; margin-bottom:12px; }
  .badge h3 { margin:0 0 6px; font-size:1.15rem; font-weight:700; color:inherit !important; }
  .badge p { margin:0; font-size:.94rem; line-height:1.6; color:inherit !important; }
  .badge .tag { display:inline-block; font-size:.72rem; letter-spacing:1.3px; text-transform:uppercase;
                font-weight:700; opacity:.85; margin-bottom:6px; color:inherit !important; }
  .badge.hot { background:#E7F8EF; border:1px solid #A7E8C6; color:#0B5D3B !important; }
  .badge.cold { background:#FFF6E5; border:1px solid #F7DFAE; color:#8A5A0B !important; }
  .mini { display:flex; gap:10px; flex-wrap:wrap; margin:4px 0 2px; }
  .mini div { flex:1 1 120px; background:#FFFFFF; border:1px solid #D1FAE5; border-radius:12px; padding:10px 12px; text-align:center; }
  .mini span { display:block; font-size:.74rem; color:#5C7A70 !important; }
  .mini strong { font-size:1.02rem; color:#065F46 !important; }
  .footer { text-align:center; color:#7D9C92 !important; font-size:.84rem; margin-top:26px; line-height:1.8; }
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="hero">
  <div class="kicker">โครงงาน AI เพื่อธุรกิจดิจิทัล · ใบงานสัปดาห์ที่ 11 → 12</div>
  <h1>🏦 AI ทำนายลูกค้าที่มีแนวโน้มตอบรับข้อเสนอเงินฝาก</h1>
  <p>ระบบช่วยฝ่ายการตลาดธนาคาร <b>จัดลำดับว่าควรโทรหาลูกค้ารายใดก่อน</b> — ใช้โมเดล Random Forest
     ที่เรียนรู้จากลูกค้าจริง {N_ROWS:,} ราย (ตอบรับเงินฝาก {RATE*100:.1f}%)<br>
     วิชา DT36822N ปัญญาประดิษฐ์เพื่อธุรกิจดิจิทัล</p>
  <div class="chips">
    <div class="chip solid">ทีม OKB EiEi</div>
    {''.join(f'<div class="chip">{c} {n}</div>' for c, n in TEAM)}
    <div class="chip">ความแม่น {ACC*100:.2f}% · AUC {AUC:.3f}</div>
  </div>
</div>
""", unsafe_allow_html=True)

tab_p, tab_m, tab_h = st.tabs(["🎯 ประเมินลูกค้า", "🧠 ข้อมูลโมเดล", "📘 วิธีใช้ & ข้อจำกัด"])

with tab_p:
    pre_mode = st.toggle("โหมดก่อนโทร (ตัด “ความยาวสายสนทนา” ออก — ใช้ข้อมูลที่มีก่อนยกหู)",
                         value=False,
                         help="ความยาวสายสนทนารู้ได้หลังคุยจบแล้ว จึงเป็นข้อมูลที่ทำให้ผลดูดีเกินจริง "
                              "ถ้าต้องเลือกว่าจะโทรหาใครก่อน ให้ใช้โหมดนี้")
    left, right = st.columns([1.05, 1], gap="large")
    with left:
        with st.form("deposit_form"):
            st.markdown("**ข้อมูลลูกค้าและประวัติการติดต่อ**")
            st.caption("กรอกตามข้อมูลในระบบ CRM ของธนาคาร — ระบบจะประเมินโอกาสที่ลูกค้าจะตอบรับเงินฝาก")
            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input(NICE_NUM["age"], int(NUMR.get("age", {}).get("min", 18)),
                                      int(NUMR.get("age", {}).get("max", 95)),
                                      int(NUMR.get("age", {}).get("median", 39)), step=1)
                balance = st.number_input(NICE_NUM["balance"], -10000, 120000,
                                          int(NUMR.get("balance", {}).get("median", 550)), step=50)
                day = st.number_input(NICE_NUM["day"], 1, 31, 15, step=1)
                if not pre_mode:
                    duration = st.number_input(NICE_NUM["duration"], 0, 4000,
                                               int(NUMR.get("duration", {}).get("median", 250)), step=10)
                else:
                    duration = int(NUMR.get("duration", {}).get("median", 250))
            with c2:
                campaign = st.number_input(NICE_NUM["campaign"], 1, 60, 1, step=1)
                pdays = st.number_input(NICE_NUM["pdays"], -1, 900, -1, step=1)
                previous = st.number_input(NICE_NUM["previous"], 0, 60, 0, step=1)
                housing = st.selectbox("มีสินเชื่อบ้าน (Housing)", ["no", "yes"], index=0)
                loan = st.selectbox("มีสินเชื่อส่วนบุคคล (Loan)", ["no", "yes"], index=0)
                default = st.selectbox("มีประวัติผิดนัดชำระ (Default)", ["no", "yes"], index=0)
            st.divider()
            c3, c4, c5 = st.columns(3)
            with c3:
                job = st.selectbox(NICE["job"], CATS.get("job", ["management"]))
                marital = st.selectbox(NICE["marital"], CATS.get("marital", ["married"]))
            with c4:
                education = st.selectbox(NICE["education"], CATS.get("education", ["secondary"]))
                contact = st.selectbox(NICE["contact"], CATS.get("contact", ["cellular"]))
            with c5:
                month = st.selectbox(NICE["month"], CATS.get("month", ["may"]))
                poutcome = st.selectbox(NICE["poutcome"], CATS.get("poutcome", ["unknown"]))
            go = st.form_submit_button("ประเมินโอกาสตอบรับเงินฝาก", type="primary", width="stretch")
    with right:
        if go:
            feats = FEATURES_PRE if pre_mode else FEATURES
            row = {f: 0 for f in feats}
            row.update({"age": float(age), "balance": float(balance), "day": float(day),
                        "campaign": float(campaign), "pdays": float(pdays),
                        "previous": float(previous), "default_yes": int(default == "yes"),
                        "housing_yes": int(housing == "yes"), "loan_yes": int(loan == "yes"),
                        "never_contacted": int(pdays < 0), "balance_negative": int(balance < 0),
                        "campaign_many": int(campaign > 3),
                        "age_group": float(0 if age <= 30 else 1 if age <= 45 else 2 if age <= 60 else 3)})
            if not pre_mode:
                row["duration"] = float(duration)
            for cat, val in (("job", job), ("marital", marital), ("education", education),
                             ("contact", contact), ("month", month), ("poutcome", poutcome)):
                key = f"{cat}_{val}"
                if key in row:
                    row[key] = 1
            X = pd.DataFrame([row], columns=feats)
            mdl = model_pre if pre_mode else model
            pred = int(mdl.predict(X)[0])
            proba = mdl.predict_proba(X)[0]
            p = float(proba[list(mdl.classes_).index(1)]) if 1 in list(mdl.classes_) else float(pred)

            st.metric("โอกาสที่ลูกค้าจะตอบรับข้อเสนอเงินฝาก", f"{p:.1%}")
            st.progress(min(max(p, 0.0), 1.0))
            if pred == 1:
                st.markdown("""
                <div class="badge hot">
                  <div class="tag">ผลการประเมิน</div>
                  <h3>✅ ควรติดต่อลูกค้ารายนี้</h3>
                  <p><b>สิ่งที่ควรทำ:</b> จัดเข้าแคมเปญโทรออกก่อน — เตรียมข้อเสนอเงินฝากระยะสั้น/ดอกเบี้ยพิเศษ
                  และบันทึกผลการคุยกลับเข้าระบบเพื่อใช้เทรนรอบถัดไป</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="badge cold">
                  <div class="tag">ผลการประเมิน</div>
                  <h3>⏳ ยังไม่ควรลงทุนโทรตอนนี้</h3>
                  <p><b>สิ่งที่ควรทำ:</b> เลื่อนไปกลุ่มหลัง หรือใช้ช่องทางต้นทุนต่ำ (อีเมล/SMS)
                  และกลับมาประเมินใหม่เมื่อมีประวัติการติดต่อครั้งใหม่ — ช่วยลดค่าใช้จ่ายของทีมเทเลมาร์เก็ตติ้ง</p>
                </div>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="mini">
              <div><span>โหมดที่ใช้</span><strong>{'ก่อนโทร' if pre_mode else 'เต็มชุด'}</strong></div>
              <div><span>จำนวนฟีเจอร์</span><strong>{len(feats)}</strong></div>
              <div><span>เกณฑ์ตัดสินใจ</span><strong>0.50</strong></div>
              <div><span>ความแม่นของโหมดนี้</span><strong>{(PRE['test_acc'] if pre_mode else ACC)*100:.2f}%</strong></div>
            </div>""", unsafe_allow_html=True)
            with st.expander("ดูค่าที่ส่งเข้าโมเดล + ฟีเจอร์ที่โมเดลให้น้ำหนักมากที่สุด"):
                st.dataframe(X.T.rename(columns={0: "ค่าที่ส่งเข้าโมเดล"}), width="stretch")
                for f, v in list((IMP_PRE if pre_mode else IMP).items())[:6]:
                    st.write(f"- {f} — **{v*100:.1f}%**")
        else:
            st.info("กรอกข้อมูลลูกค้าด้านซ้าย แล้วกด **ประเมินโอกาสตอบรับเงินฝาก** เพื่อดูผล")
            st.caption("ระบบจะแสดง % โอกาสตอบรับ พร้อมข้อเสนอว่าควรจัดลำดับการติดต่ออย่างไร")

with tab_m:
    st.markdown("#### สรุปโมเดลสุดท้ายของทีม")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ชนิดโมเดล", "Random Forest")
    m2.metric("จำนวนฟีเจอร์", f"{len(FEATURES)}")
    m3.metric("ความแม่น (ชุดทดสอบ)", f"{ACC*100:.2f}%")
    m4.metric("ROC-AUC", f"{AUC:.3f}")
    left, right = st.columns([1, 1], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("**ตัวชี้วัดของโมเดล (คลาส “ตอบรับเงินฝาก”)**")
            st.write(f"- precision **{PREC:.3f}** · recall **{REC:.3f}** · F1 **{F1:.3f}**")
            st.write(f"- cross-validation 3-fold = **{CV3:.4f}**")
            st.write(f"- ลูกค้าตอบรับในข้อมูลจริง **{RATE*100:.1f}%** (คลาสค่อนข้างสมดุล จึงดู accuracy ควบคู่ F1 ได้)")
        with st.container(border=True):
            st.markdown("**เทียบกับงานสัปดาห์ที่ 11 ของทีม**")
            st.write(f"- Decision Tree (depth 5) ชุดทดสอบ = **{DT['test_acc']*100:.2f}%** (F1 {DT['f1']:.3f}) — โมเดลที่ทีมเลือกเพราะอธิบายง่าย")
            st.write(f"- Deep MLP (32-16, dropout .2) ที่ทีมรายงาน = **81.25%**")
            st.write(f"- Random Forest ที่ปรับ hyperparameter แล้ว = **{ACC*100:.2f}%** → ดีขึ้น **{(ACC-0.8250)*100:+.2f} จุด** จาก DT เดิม")
    with right:
        with st.container(border=True):
            st.markdown("**การทดลองคัดฟีเจอร์ด้วย GA**")
            st.write(f"- ใช้ทั้ง **{meta.get('n_features_all')} ฟีเจอร์**: CV = **{meta.get('baseline_cv3'):.4f}**")
            st.write(f"- GA คัดเหลือ **{meta.get('ga_n_selected')} ฟีเจอร์**: CV = **{meta.get('ga_cv3'):.4f}**")
            st.write("- สรุป: GA ลดฟีเจอร์ได้มาก แต่ความแม่นแทบไม่ต่าง ทีมจึงใช้ชุดฟีเจอร์ที่อธิบายให้ฝ่ายการตลาดเข้าใจได้")
        with st.container(border=True):
            st.markdown("**โหมดก่อนโทร (ตัด duration ออก)**")
            st.write(f"- ความแม่นลดจาก **{ACC*100:.2f}%** เหลือ **{PRE['test_acc']*100:.2f}%** (AUC {PRE['roc_auc']:.3f})")
            st.write("- แปลว่า “ความยาวสายสนทนา” เป็นตัวแปรที่ทรงอิทธิพลที่สุด (41.7%) "
                     "แต่รู้ได้หลังคุยจบ → ใช้ได้กับงานจัดลำดับลูกค้าหลังโทร ไม่ใช่การเลือกก่อนโทร")
    with st.container(border=True):
        st.markdown("**ฟีเจอร์ที่มีอิทธิพลมากที่สุด (โมเดลเต็มชุด)**")
        cols = st.columns(2)
        items = list(IMP.items())[:10]
        for i, (f, v) in enumerate(items):
            cols[i % 2].write(f"- {f} — **{v*100:.1f}%**")

with tab_h:
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        with st.container(border=True):
            st.markdown("#### วิธีใช้")
            st.markdown(
                "1. เลือก **โหมดก่อนโทร** ถ้าต้องตัดสินใจว่าจะโทรหาลูกค้ารายใดก่อน (ตัดความยาวสายสนทนาออก)\n"
                "2. กรอก **ข้อมูลลูกค้า**: อายุ ยอดเงินในบัญชี วันที่ติดต่อ จำนวนครั้งที่ติดต่อ ฯลฯ\n"
                "3. เลือก **อาชีพ / สถานภาพ / การศึกษา / ช่องทางติดต่อ / เดือน / ผลแคมเปญครั้งก่อน**\n"
                "4. กด **ประเมินโอกาสตอบรับเงินฝาก** → ดู % โอกาสและข้อเสนอการจัดลำดับ")
    with c2:
        with st.container(border=True):
            st.markdown("#### ข้อจำกัดของโมเดล")
            st.markdown(
                "1. **duration** (ความยาวสายสนทนา) เป็นข้อมูลที่รู้หลังคุยจบ — ถ้าใช้จะได้ความแม่นสูงถึง 85.45% "
                "แต่ไม่สามารถใช้ตัดสินใจก่อนโทรได้จริง (โหมดก่อนโทรได้ 73.13%)\n"
                "2. ข้อมูลเป็นแคมเปญโทรของธนาคารโปรตุเกส (2008–2010) — พฤติกรรมลูกค้าปัจจุบันอาจต่างกัน ควรทดสอบกับข้อมูลจริงก่อนใช้\n"
                "3. โมเดลให้ผลเป็น **ความน่าจะเป็น** ไม่ใช่คำสั่ง ควรใช้ร่วมกับดุลพินิจของเจ้าหน้าที่และข้อจำกัดด้านกฎหมายคุ้มครองข้อมูล\n"
                "4. ควรติดตามผลจริง (conversion rate) ของแต่ละกลุ่มที่โทร เพื่อวัดว่าโมเดลช่วยลดต้นทุนได้จริงหรือไม่")

st.markdown(f"""
<div class="footer">
  จัดทำโดย <b>ทีม OKB EiEi</b> — {' · '.join(f'{c} {n}' for c, n in TEAM)}<br>
  วิชา DT36822N ปัญญาประดิษฐ์เพื่อธุรกิจดิจิทัล · ใบงานสัปดาห์ที่ 11-12 (Streamlit) ·
  ข้อมูล: Bank Marketing Dataset (bank.csv) จำนวน {N_ROWS:,} ราย
</div>
""", unsafe_allow_html=True)
