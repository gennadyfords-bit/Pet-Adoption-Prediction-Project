"""
Калькулятор шансов пристройства питомца (Pet Adoption Prediction).
Использует модель RandomForestClassifier из classification/models-comparison.
Запуск: streamlit run app.py (из папки s-streamlit) или streamlit run s-streamlit/app.py (из корня).
"""

import numpy as np
import streamlit as st

from config import FEATURE_COLUMNS, ADOPTION_SPEED_LABELS, MODEL_PATH

st.set_page_config(page_title="Калькулятор шансов пристройства", layout="centered")
st.title("Pet Adoption Prediction — Калькулятор шансов")
st.caption("Оценка скорости пристройства питомца по признакам объявления (модель RandomForest).")

# Загрузка модели
@st.cache_resource
def load_model():
    try:
        import joblib
        with open(MODEL_PATH, "rb") as f:
            return joblib.load(f)
    except FileNotFoundError:
        st.error(
            f"Файл модели не найден: {MODEL_PATH}. "
            "Сохраните обученную модель из ноутбука classification в этот файл (model_rf_baseline.pkl)."
        )
        return None

model = load_model()
if model is None:
    st.stop()

# Форма ввода (упрощённые границы; при необходимости подстройте по EDA)
st.subheader("Параметры объявления")

col1, col2 = st.columns(2)
with col1:
    type_ = st.number_input("Type (1=Dog, 2=Cat)", min_value=1, max_value=2, value=1, step=1)
    age = st.number_input("Age (месяцев)", min_value=0, max_value=255, value=12, step=1)
    breed1 = st.number_input("Breed1", min_value=0, max_value=307, value=0, step=1)
    breed2 = st.number_input("Breed2", min_value=0, max_value=307, value=0, step=1)
    gender = st.number_input("Gender (1=Male, 2=Female, 3=Mixed)", min_value=1, max_value=3, value=1, step=1)
    color1 = st.number_input("Color1", min_value=0, max_value=7, value=1, step=1)
    color2 = st.number_input("Color2", min_value=0, max_value=7, value=0, step=1)
    maturity_size = st.number_input("MaturitySize (1–4)", min_value=1, max_value=4, value=2, step=1)

with col2:
    fur_length = st.number_input("FurLength (1–3)", min_value=1, max_value=3, value=1, step=1)
    health = st.number_input("Health (0–3)", min_value=0, max_value=3, value=0, step=1)
    quantity = st.number_input("Quantity", min_value=1, max_value=10, value=1, step=1)
    state = st.number_input("State", min_value=0, max_value=15, value=0, step=1)
    video_amt = st.number_input("VideoAmt", min_value=0, max_value=10, value=0, step=1)
    photo_amt = st.number_input("PhotoAmt", min_value=0, max_value=20, value=1, step=1)
    medical_check = st.number_input("MedicalCheck (0/1)", min_value=0, max_value=1, value=0, step=1)
    is_paid = st.number_input("IsPaid (0/1)", min_value=0, max_value=1, value=0, step=1)

# Вектор в порядке FEATURE_COLUMNS
X = np.array([[
    type_, age, breed1, breed2, gender, color1, color2, maturity_size,
    fur_length, health, quantity, state, video_amt, photo_amt, medical_check, is_paid
]])

if st.button("Рассчитать шансы"):
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    st.subheader("Результат")
    st.success(f"**Ожидаемая скорость пристройства:** {ADOPTION_SPEED_LABELS.get(pred, pred)} (класс {pred})")

    st.write("**Вероятности по исходам:**")
    for i, label in ADOPTION_SPEED_LABELS.items():
        pct = proba[i] * 100
        st.progress(float(proba[i]), text=f"{label}: {pct:.1f}%")
