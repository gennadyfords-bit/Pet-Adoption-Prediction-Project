import itertools
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


@dataclass
class RFConfig:
    n_estimators: int
    max_depth: int | None
    min_samples_split: int
    min_samples_leaf: int
    criterion: str = "gini"
    random_state: int = 42


def load_data(csv_path: Path):
    """Загружаем данные и делим на train/valid.

    Ожидается, что в датасете целевая переменная называется 'AdoptionSpeed'.
    Ненумерические поля (`Name`, `Description`, `RescuerID`, `PetID`) для простоты отбрасываются.
    """
    df = pd.read_csv(csv_path)

    if "AdoptionSpeed" not in df.columns:
        raise ValueError("В датасете нет столбца 'AdoptionSpeed' – проверьте входные данные.")

    y = df["AdoptionSpeed"]

    # Используем только числовые признаки (как в описании датасета)
    X = df.select_dtypes(include=[np.number]).drop(columns=["AdoptionSpeed"])

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    return X_train, X_valid, y_train, y_valid


def init_db(db_path: Path):
    """Создаём (при необходимости) таблицу для логов в SQLite."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS rf_experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            n_estimators INTEGER NOT NULL,
            max_depth INTEGER,
            min_samples_split INTEGER NOT NULL,
            min_samples_leaf INTEGER NOT NULL,
            criterion TEXT NOT NULL,
            random_state INTEGER NOT NULL,
            f1_weighted REAL NOT NULL,
            accuracy REAL NOT NULL,
            train_time_sec REAL NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    conn.commit()
    return conn


def log_run(conn: sqlite3.Connection, cfg: RFConfig, f1_w: float, acc: float, train_time: float):
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO rf_experiments (
            model,
            n_estimators,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            criterion,
            random_state,
            f1_weighted,
            accuracy,
            train_time_sec,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "RandomForestClassifier",
            cfg.n_estimators,
            cfg.max_depth,
            cfg.min_samples_split,
            cfg.min_samples_leaf,
            cfg.criterion,
            cfg.random_state,
            float(f1_w),
            float(acc),
            float(train_time),
            datetime.utcnow().isoformat(timespec="seconds"),
        ),
    )
    conn.commit()


def run_manual_search(
    csv_path: str = "train.csv",
    db_path: str = "rf_experiments.db",
):
    csv_path = Path(csv_path)
    db_path = Path(db_path)

    X_train, X_valid, y_train, y_valid = load_data(csv_path)
    conn = init_db(db_path)

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "criterion": ["gini", "entropy"],
    }

    keys = list(param_grid.keys())
    best_f1 = -np.inf
    best_cfg: RFConfig | None = None

    print("Запускаем ручной перебор параметров RandomForestClassifier...")

    for values in itertools.product(*(param_grid[k] for k in keys)):
        params = dict(zip(keys, values))
        cfg = RFConfig(**params)

        rf = RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            min_samples_split=cfg.min_samples_split,
            min_samples_leaf=cfg.min_samples_leaf,
            criterion=cfg.criterion,
            random_state=cfg.random_state,
            n_jobs=-1,
        )

        start = perf_counter()
        rf.fit(X_train, y_train)
        train_time = perf_counter() - start

        y_pred = rf.predict(X_valid)
        f1_w = f1_score(y_valid, y_pred, average="weighted")
        acc = accuracy_score(y_valid, y_pred)

        log_run(conn, cfg, f1_w, acc, train_time)

        print(
            "Параметры:",
            asdict(cfg),
            f"--> F1_weighted={f1_w:.4f}, Accuracy={acc:.4f}, time={train_time:.2f}s",
        )

        if f1_w > best_f1:
            best_f1 = f1_w
            best_cfg = cfg

    conn.close()

    if best_cfg is not None:
        print("\nЛучший результат по weighted F1:")
        print(asdict(best_cfg))
        print(f"F1_weighted={best_f1:.4f}")


if __name__ == "__main__":
    # По умолчанию ожидаем файл `train.csv` в текущей директории (`classification/`).
    # При необходимости можно передать другие пути через аргументы или изменить значения ниже.
    run_manual_search(csv_path="train.csv", db_path="rf_experiments.db")

