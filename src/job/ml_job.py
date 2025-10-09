import os
import json
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from db_querys.etl_producao import (
    extrair_dados_producao,
    transformar_producao,
    carregar_em_dataset_ml as carregar_producao,
)
from db_querys.etl_exportacao import (
    extrair_dados_exportacao,
    transformar_exportacao,
    carregar_em_dataset_ml as carregar_exportacao,
)
from db_querys import init_db  # já existente

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@db:5432/projeto3")


# -------------------- persistência (dataset e métricas) --------------------
def _ensure_metrics_table(engine):
    ddl = """
    CREATE TABLE IF NOT EXISTS ml_metrics (
        id SERIAL PRIMARY KEY,
        accuracy DOUBLE PRECISION NOT NULL,
        confusion_matrix TEXT NOT NULL,
        classification_report TEXT NOT NULL,
        rows INT NOT NULL,
        created_at TIMESTAMP NOT NULL
    );
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


def _save_ml_dataset(engine, df_ds: pd.DataFrame):
    with engine.begin() as conn:
        df_ds.to_sql("ml_dataset", con=conn, if_exists="replace", index=False)


def _save_ml_metrics(
    engine, accuracy: float, cm: List[List[int]], report: dict, rows: int
):
    _ensure_metrics_table(engine)
    payload = {
        "accuracy": accuracy,
        "confusion_matrix": json.dumps(cm, ensure_ascii=False),
        "classification_report": json.dumps(report, ensure_ascii=False),
        "rows": int(rows),
        "created_at": datetime.utcnow(),
    }
    with engine.begin() as conn:
        conn.execute(
            text(
                """
            INSERT INTO ml_metrics
            (accuracy, confusion_matrix, classification_report, rows, created_at)
            VALUES (:accuracy, :confusion_matrix, :classification_report, :rows, :created_at)
        """
            ),
            payload,
        )


def _ensure_etl_columns(engine):
    """Garante etl_job e colunas (sem replace)."""
    with engine.begin() as conn:
        conn.execute(
            text(
                """
            CREATE TABLE IF NOT EXISTS etl_job (
                year INT PRIMARY KEY,
                producao_vinhos_mesa DOUBLE PRECISION
            );
        """
            )
        )
        conn.execute(
            text(
                """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name='etl_job' AND column_name='exportacao_total_dols'
                ) THEN
                    ALTER TABLE etl_job ADD COLUMN exportacao_total_dols DOUBLE PRECISION;
                END IF;

                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name='etl_job' AND column_name='desempenho'
                ) THEN
                    ALTER TABLE etl_job ADD COLUMN desempenho INT;
                END IF;
            END$$;
        """
            )
        )


# -------------------- regra ANTIGA (média móvel 5 anos) --------------------
def _classificar_regra_antiga(df_etl: pd.DataFrame) -> pd.DataFrame:
    """
    Regras:
      - prod > média & exp > média → 1
      - prod > média & exp < média → 0
      - prod < média & exp > média → 2
      - prod < média & exp < média → 1
    Usa média móvel (5) dos ANOS ANTERIORES (shift).
    """
    df = df_etl.sort_values("year").reset_index(drop=True).copy()

    df["prod_media_5"] = (
        df["producao_vinhos_mesa"].shift(1).rolling(window=5, min_periods=3).mean()
    )
    df["exp_media_5"] = (
        df["exportacao_total_dols"].shift(1).rolling(window=5, min_periods=3).mean()
    )

    df = df.dropna(subset=["prod_media_5", "exp_media_5"]).copy()

    def _label(r) -> int:
        prod_maior = r["producao_vinhos_mesa"] > r["prod_media_5"]
        exp_maior = r["exportacao_total_dols"] > r["exp_media_5"]
        if prod_maior and exp_maior:
            return 1
        elif prod_maior and not exp_maior:
            return 0
        elif (not prod_maior) and exp_maior:
            return 2
        else:
            return 1

    df["desempenho"] = df.apply(_label, axis=1).astype(int)
    return df[["year", "producao_vinhos_mesa", "exportacao_total_dols", "desempenho"]]


def run_and_save_ml() -> None:
    """
    - Executa ETLs de exportação e produção (para dataset_ml)
    - Lê dataset_ml, classifica via regra antiga e grava 'desempenho' (UPDATE por ano)
    - Monta ml_dataset e treina regressão logística
    - Salva métricas em ml_metrics
    """
    engine = create_engine(DATABASE_URL, future=True, pool_pre_ping=True)
    session = init_db()

    # 1) Executa os ETLs para a tabela dataset_ml
    intervalo = (1980, 2023)

    # Produção
    df_prod = transformar_producao(extrair_dados_producao(session, intervalo))
    carregar_producao(session, df_prod)

    # Exportação
    df_exp = transformar_exportacao(
        extrair_dados_exportacao(session, tab="Exportações", intervalo=intervalo)
    )
    carregar_exportacao(session, df_exp)

    # 2) lê dataset_ml
    with engine.begin() as conn:
        df = pd.read_sql("SELECT * FROM dataset_ml ORDER BY ano", conn)

    needed = {"ano", "producao_litros", "exportacao_total"}
    if df.empty or not needed.issubset(df.columns):
        return

    df["producao_litros"] = pd.to_numeric(df["producao_litros"], errors="coerce")
    df["exportacao_total"] = pd.to_numeric(df["exportacao_total"], errors="coerce")
    df = df.dropna(subset=["producao_litros", "exportacao_total"])

    # 3) classificar
    df_sorted = df.sort_values("ano").copy()
    df_sorted["prod_media_5"] = (
        df_sorted["producao_litros"].shift(1).rolling(window=5, min_periods=3).mean()
    )
    df_sorted["exp_media_5"] = (
        df_sorted["exportacao_total"].shift(1).rolling(window=5, min_periods=3).mean()
    )
    df_sorted = df_sorted.dropna(subset=["prod_media_5", "exp_media_5"]).copy()

    def _label(r) -> int:
        prod_maior = r["producao_litros"] > r["prod_media_5"]
        exp_maior = r["exportacao_total"] > r["exp_media_5"]
        if prod_maior and exp_maior:
            return 1
        elif prod_maior and not exp_maior:
            return 0
        elif (not prod_maior) and exp_maior:
            return 2
        else:
            return 1

    df_sorted["desempenho"] = df_sorted.apply(_label, axis=1).astype(int)

    # 4) atualiza dataset_ml com desempenho
    with engine.begin() as conn:
        for _, r in df_sorted.iterrows():
            conn.execute(
                text("UPDATE dataset_ml SET desempenho=:d WHERE ano=:y"),
                {"y": int(r["ano"]), "d": int(r["desempenho"])},
            )

    # 5) preparar dataset de treino
    ds = df_sorted.rename(
        columns={
            "ano": "year",
            "producao_litros": "producao_total",
            "exportacao_total": "exportacao_total_usd",
        }
    )[["year", "producao_total", "exportacao_total_usd", "desempenho"]]

    if ds.empty or len(ds) < 6:
        _save_ml_dataset(engine, ds)
        return

    _save_ml_dataset(engine, ds)

    # 6) treino e avaliação
    X = ds[["year", "producao_total", "exportacao_total_usd"]].values
    y = ds["desempenho"].values

    classes, counts = np.unique(y, return_counts=True)
    if classes.size < 2:
        print(
            f"[ML] Abortado: apenas uma classe presente em y={classes.tolist()} counts={counts.tolist()}"
        )
        return

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=42
        )

    if np.unique(y_train).size < 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.8, random_state=42
        )
        if np.unique(y_train).size < 2:
            print("[ML] Abortado: treino com 1 classe mesmo após novo split.")
            return

    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2]).tolist()
    cr = classification_report(y_test, y_pred, labels=[0, 1, 2], output_dict=True)

    _save_ml_metrics(engine, acc, cm, cr, rows=len(ds))
