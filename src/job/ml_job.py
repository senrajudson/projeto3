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

from db_querys.etl_producao import extrair_dados_producao, transformar_producao, carregar_em_dataset_ml as carregar_producao
from db_querys.etl_exportacao import extrair_dados_exportacao, transformar_exportacao, carregar_em_dataset_ml as carregar_exportacao
from db_querys.db_utils import init_db

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@db:5432/projeto3")


def _ensure_metrics_table(engine):
    """Cria tabela de métricas de modelo se não existir."""
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


def _ensure_predictions_table(engine):
    """Cria tabela de comparação entre real e previsto."""
    ddl = """
    CREATE TABLE IF NOT EXISTS ml_predictions_vs_real (
        year INT PRIMARY KEY,
        real INT NOT NULL,
        predicted INT NOT NULL
    );
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


def _save_ml_dataset(engine, df_ds: pd.DataFrame):
    """Salva o dataset final usado para treino do modelo."""
    with engine.begin() as conn:
        df_ds.to_sql("ml_dataset", con=conn, if_exists="replace", index=False)


def _save_ml_metrics(engine, accuracy: float, cm: List[List[int]], report: dict, rows: int):
    """Salva as métricas do modelo treinado."""
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
            text("""
            INSERT INTO ml_metrics (accuracy, confusion_matrix, classification_report, rows, created_at)
            VALUES (:accuracy, :confusion_matrix, :classification_report, :rows, :created_at)
            """),
            payload
        )


def _aplicar_regra_desempenho(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica regra de classificação com base em média móvel 5 anos.
    Regras:
        - prod > média & exp > média → 1
        - prod > média & exp < média → 0
        - prod < média & exp > média → 2
        - prod < média & exp < média → 1
    """
    df_sorted = df.sort_values("ano").copy()
    df_sorted["prod_media_5"] = df_sorted["producao_litros"].shift(1).rolling(5, min_periods=3).mean()
    df_sorted["exp_media_5"] = df_sorted["exportacao_total"].shift(1).rolling(5, min_periods=3).mean()
    df_sorted = df_sorted.dropna(subset=["prod_media_5", "exp_media_5"])

    def _regra(row):
        p_maior = row["producao_litros"] > row["prod_media_5"]
        e_maior = row["exportacao_total"] > row["exp_media_5"]
        if p_maior and e_maior:
            return 1
        elif p_maior:
            return 0
        elif e_maior:
            return 2
        return 1

    df_sorted["desempenho"] = df_sorted.apply(_regra, axis=1).astype(int)
    return df_sorted


def run_and_save_ml():
    """Roda o pipeline de ETL + classificação + treino do modelo + persistência."""
    engine = create_engine(DATABASE_URL, future=True, pool_pre_ping=True)
    session = init_db()
    _ensure_predictions_table(engine)

    # Executa ETLs
    intervalo = (1980, 2023)
    df_prod = transformar_producao(extrair_dados_producao(session, intervalo))
    carregar_producao(session, df_prod)

    df_exp = transformar_exportacao(
        extrair_dados_exportacao(session, tab="Exportacao", intervalo=intervalo)
    )
    carregar_exportacao(session, df_exp)

    # Carrega dataset bruto
    with engine.begin() as conn:
        df = pd.read_sql("SELECT * FROM dataset_ml ORDER BY ano", conn)

    if df.empty or not {"ano", "producao_litros", "exportacao_total"}.issubset(df.columns):
        return

    df["producao_litros"] = pd.to_numeric(df["producao_litros"], errors="coerce")
    df["exportacao_total"] = pd.to_numeric(df["exportacao_total"], errors="coerce")
    df = df.dropna(subset=["producao_litros", "exportacao_total"])

    # Classifica via regra antiga
    df_desemp = _aplicar_regra_desempenho(df)

    # Atualiza valores no dataset_ml
    with engine.begin() as conn:
        for _, r in df_desemp.iterrows():
            conn.execute(
                text("UPDATE dataset_ml SET desempenho=:d WHERE ano=:y"),
                {"y": int(r["ano"]), "d": int(r["desempenho"])}
            )

    # Monta dataset final
    ds = df_desemp.rename(columns={
        "ano": "year",
        "producao_litros": "producao_total",
        "exportacao_total": "exportacao_total_usd"
    })[["year", "producao_total", "exportacao_total_usd", "desempenho"]]

    if ds.empty or len(ds) < 6:
        _save_ml_dataset(engine, ds)
        return

    _save_ml_dataset(engine, ds)

    # Treino do modelo
    X = ds[["year", "producao_total", "exportacao_total_usd"]].values
    y = ds["desempenho"].values

    if np.unique(y).size < 2:
        print("[ML] Abortado: apenas uma classe presente")
        return

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    if np.unique(y_train).size < 2:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
        if np.unique(y_train).size < 2:
            print("[ML] Abortado: treino com 1 classe mesmo após novo split.")
            return

    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Salva comparação real vs previsto
    df_pred = pd.DataFrame({
        "year": X_test[:, 0].astype(int),
        "real": y_test.astype(int),
        "predicted": y_pred.astype(int)
    })
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM ml_predictions_vs_real"))
        df_pred.to_sql("ml_predictions_vs_real", con=conn, if_exists="append", index=False)

    # Salva métricas
    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2]).tolist()
    cr = classification_report(y_test, y_pred, labels=[0, 1, 2], output_dict=True)
    _save_ml_metrics(engine, acc, cm, cr, rows=len(ds))
