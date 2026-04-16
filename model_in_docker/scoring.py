#1. Чтение данных
#2. Обработка данных
#3. Применить модель
#4. Сохранить скоры
# Путь к .env строится от расположения файла, а не от текущей рабочей директории терминала.
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.abspath(os.path.join(_dir, "..", ".env"))
if load_dotenv(dotenv_path):
    print("Файл .env успешно загружен")
else:
    print(f"Не удалось найти файл .env по пути: {dotenv_path}")

# Учётные данные и строка подключения Oracle (значения из .env).
USER = os.getenv('DB_USER')
PASSWORD = os.getenv('DB_PASSWORD')
HOST = os.getenv('DB_HOST')
PORT = os.getenv('DB_PORT')
SERVICE = os.getenv('DB_SERVICE')

conn_str = f"oracle+oracledb://{USER}:{PASSWORD}@{HOST}:{PORT}/?service_name={SERVICE}"

try:
    engine = create_engine(conn_str)
    print("Подключение к базе данных настроено")
except Exception as e:
    print(f"Ошибка при создании engine: {e}")

# Исходная таблица для feature engineering.
query = "SELECT * FROM ANIMAL_INFORMATION_FEATURE_ENGINEERING"

try:
    df_feature_engineering = pd.read_sql(query, engine)
    print(f"Данные успешно загружены! Размер таблицы: {df_feature_engineering.shape}")
except Exception as e:
    print(f"Ошибка при выполнении запроса: {e}")

# 1. Заменяем пропуски в колонке Description
df_feature_engineering['Description'] = df_feature_engineering['Description'].fillna("No description provided")

# 2. Заменяем пропуски в колонке Name
df_feature_engineering['Name'] = df_feature_engineering['Name'].fillna("Unnamed")

# Создаем функцию для перевода в баллы
# 1 (Да) -> 1 балл
# 2 (Нет) и 3 (Не уверен) -> 0 баллов
def get_medical_score(val):
    return 1 if val == 1 else 0

# Применяем трансформацию и складываем результаты
df_feature_engineering['MedicalCheck'] = (
    df_feature_engineering['Vaccinated'].apply(get_medical_score) +
    df_feature_engineering['Dewormed'].apply(get_medical_score) +
    df_feature_engineering['Sterilized'].apply(get_medical_score)
)

# Создаем признак IsPaid: 1 если Fee > 0, иначе 0
df_feature_engineering['IsPaid'] = (df_feature_engineering['Fee'] > 0).astype(int)

# Удаляем исходные признаки из датафрейма
df_feature_engineering = df_feature_engineering.drop(columns=['Vaccinated', 'Dewormed', 'Sterilized', 'Fee'])

# Удаляем исходный признак из датафрейма
df_feature_engineering = df_feature_engineering.drop(columns=['Color3'])
# Выборка из целевой/ML-таблицы (первая загрузка).
query = "SELECT * FROM ANIMAL_INFORMATION_ML"

try:
    df_ml = pd.read_sql(query, engine)
    print(f"Данные успешно загружены! Размер таблицы: {df_ml.shape}")
except Exception as e:
    print(f"Ошибка при выполнении запроса: {e}")
target_table_name = 'animal_information_ml' # используем нижний регистр

# Запись обработанного датафрейма в БД; при существующей таблице — replace.
try:
    df_feature_engineering.to_sql(
        name=target_table_name, 
        con=engine, 
        if_exists='replace', 
        index=False
    )
    print(f"Таблица {target_table_name} успешно создана/обновлена.")
except Exception as e:
    print(f"Ошибка: {e}")

# Повторная загрузка той же ML-таблицы (как в исходном скрипте).
query = "SELECT * FROM ANIMAL_INFORMATION_ML"

try:
    df_ml = pd.read_sql(query, engine)
    print(f"Данные успешно загружены! Размер таблицы: {df_ml.shape}")
except Exception as e:
    print(f"Ошибка при выполнении запроса: {e}")

# --- Обучение CatBoost: целевая переменная AdoptionSpeed, loss MultiClass, оценка по WKappa ---
y_cb = df_ml["AdoptionSpeed"]
X_cb = df_ml.select_dtypes(include=[np.number]).drop(columns=["AdoptionSpeed"], errors="ignore")

X_train_cb, X_valid_cb, y_train_cb, y_valid_cb = train_test_split(
    X_cb,
    y_cb,
    test_size=0.2,
    random_state=42,
    stratify=y_cb,
)

train_pool_cb = Pool(X_train_cb, y_train_cb)
valid_pool_cb = Pool(X_valid_cb, y_valid_cb)

catboost_model = CatBoostClassifier(
    loss_function="MultiClass",
    eval_metric="WKappa",
    random_seed=42,
    verbose=100,
    iterations=2000,
    early_stopping_rounds=100,
    use_best_model=True,
)

catboost_model.fit(train_pool_cb, eval_set=valid_pool_cb)

best_scores = catboost_model.get_best_score()
print("CatBoost: лучшие значения метрик на валидации:", best_scores)
val_wkappa = best_scores.get("validation", {}).get("WKappa")
if val_wkappa is not None:
    print(f"WKappa (validation): {val_wkappa:.6f}")

# --- Скоры по всему df_ml → pet_adoption_scores.csv в текущей рабочей директории (cwd) ---
X_score = df_ml.reindex(columns=X_cb.columns)
pred_cb = catboost_model.predict(X_score).astype(int).ravel()
proba_cb = catboost_model.predict_proba(X_score)
classes_cb = [int(c) for c in catboost_model.classes_]

score_parts = []
if "PetID" in df_ml.columns:
    score_parts.append(df_ml[["PetID"]].reset_index(drop=True))
else:
    score_parts.append(pd.DataFrame({"row_index": np.arange(len(df_ml))}))
score_parts.append(pd.DataFrame({"predicted_adoption_speed": pred_cb}))
for i, c in enumerate(classes_cb):
    score_parts.append(pd.DataFrame({f"prob_{c}": proba_cb[:, i]}))
scores_df = pd.concat(score_parts, axis=1)

scores_path = os.path.join(os.getcwd(), "pet_adoption_scores.csv")
scores_df.to_csv(scores_path, index=False)
print(f"Скоры сохранены: {scores_path} ({len(scores_df)} строк)")
