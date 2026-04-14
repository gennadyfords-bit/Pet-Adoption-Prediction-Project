# Калькулятор шансов пристройства (Streamlit)

Приложение для оценки скорости пристройства питомца по модели RandomForest из `classification`.

## Подготовка

1. **Модель.** Сохраните обученную модель в эту папку. В ноутбуке `classification` (после ячейки с `rf_best.fit(X, y)` и регистрацией в MLflow) выполните:

   ```python
   import joblib
   import os
   joblib.dump(rf_best, os.path.join("..", "s-streamlit", "model_rf_baseline.pkl"))
   ```

   Либо скопируйте `model.pkl` из  
   `classification/mlruns/667762689781677551/models/m-077c0c0a24474b14952ac7ceb2f0e100/artifacts/model.pkl`  
   в `s-streamlit/model_rf_baseline.pkl`.

2. **Зависимости:**

   ```bash
   pip install -r s-streamlit/requirements.txt
   ```

## Как запустить калькулятор

1. Откройте репозиторий **Pet-Adoption-Prediction-Project**.
2. Откройте ноутбук **models-comparison-checkpoint.ipynb** (в папке `classification/.ipynb_checkpoints/`).
3. Запустите код в ноутбуке (чтобы модель экспортировалась в `s-streamlit`, если ещё не экспортирована).
4. Перейдите в терминал.
5. Убедитесь, что вы в корне проекта (например, `~/Pet-Adoption-Prediction-Project`, ветка `s-streamlit-test`), и выполните команду:

   ```bash
   streamlit run s-streamlit/app.py
   ```

## Файлы в папке

| Файл | Назначение |
|------|------------|
| `app.py` | Приложение Streamlit (форма ввода, расчёт, вывод шансов) |
| `config.py` | Путь к модели, список признаков, подписи AdoptionSpeed |
| `requirements.txt` | Зависимости для запуска |
| `model_rf_baseline.pkl` | Модель (создаётся вручную при экспорте из ноутбука) |
| `STREAMLIT_PLAN.md` | План и чек-лист |
