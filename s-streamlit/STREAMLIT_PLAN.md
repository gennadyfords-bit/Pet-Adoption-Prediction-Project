# План: калькулятор шансов пристройства питомца (Streamlit)

Все файлы Streamlit — в папке **`s-streamlit`**.

## Контекст
- Модель: **RandomForestClassifier** (выбрана после сравнения в `classification/.../models-comparison-checkpoint.ipynb`).
- Целевая переменная: **AdoptionSpeed** (0–4: чем меньше — тем быстрее пристройство).
- Признаки: Type, Age, Breed1, Breed2, Gender, Color1, Color2, MaturitySize, FurLength, Health, Quantity, State, VideoAmt, PhotoAmt, MedicalCheck, IsPaid.

---

## Последовательность действий

### 1. Подготовка модели для Streamlit
- Сохранить обученную модель в `s-streamlit/model_rf_baseline.pkl` (из ноутбука в папке `classification` — добавить ячейку с `joblib.dump(rf_best, "../s-streamlit/model_rf_baseline.pkl")` или скопировать из MLflow artifacts).
- Список признаков задаётся в `config.py` / в `app.py`.

### 2. Создание приложения Streamlit
- Установить зависимости: `pip install -r s-streamlit/requirements.txt`.
- Запуск из корня проекта: `streamlit run s-streamlit/app.py` или из папки: `cd s-streamlit && streamlit run app.py`.

### 3. Форма ввода (калькулятор)
- В `app.py`: для каждого из 16 признаков — `st.number_input` или слайдер с min/max по доменам.
- Ввод собирается в вектор в порядке признаков и передаётся в `model.predict()` и `model.predict_proba()`.

### 4. Вывод результата (шансы)
- Предсказанный класс (0–4) и подписи: тот же день / 1–7 дней / 8–30 дней / 31–90 дней / не пристроено за 100 дней.
- Вероятности по классам — как проценты («шансы»).

### 5. Запуск и проверка
- Из корня: `streamlit run s-streamlit/app.py`.
- Проверить порядок признаков и типы.

### 6. Чек-лист
1. [ ] Сохранить модель в `s-streamlit/model_rf_baseline.pkl`.
2. [ ] Запустить приложение из `s-streamlit`.
3. [ ] Проверить вывод предсказания и вероятностей.
