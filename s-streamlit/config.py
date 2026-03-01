# Конфигурация калькулятора шансов пристройства (Pet Adoption Prediction)
# Порядок признаков должен совпадать с обучением в models-comparison-checkpoint.ipynb

import os

# Путь к модели относительно папки s-streamlit
DIR_APP = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(DIR_APP, "model_rf_baseline.pkl")

# Признаки в том же порядке, что при обучении (числовые столбцы без AdoptionSpeed)
FEATURE_COLUMNS = [
    "Type",
    "Age",
    "Breed1",
    "Breed2",
    "Gender",
    "Color1",
    "Color2",
    "MaturitySize",
    "FurLength",
    "Health",
    "Quantity",
    "State",
    "VideoAmt",
    "PhotoAmt",
    "MedicalCheck",
    "IsPaid",
]

# Подписи для AdoptionSpeed (0–4)
ADOPTION_SPEED_LABELS = {
    0: "В тот же день",
    1: "В первую неделю (1–7 дней)",
    2: "В первый месяц (8–30 дней)",
    3: "В первые 3 месяца (31–90 дней)",
    4: "Не пристроено за 100 дней",
}
