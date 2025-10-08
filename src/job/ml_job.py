# src/ml_job.py
import os
import json
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# === módulos ETL novos ===
from job.etl_job1 import run_etl_save
from job.etl_job2 import run_etl_export_update

# =========================================================
# Config
# =========================================================
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@db:5432/projeto3")


# =========================================================
# Persistência (dataset e métricas)
# =========================================================
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
    """
    Garante que a tabela etl_job existe e possui as colunas necessárias,
    incluindo 'desempenho' para gravar a classificação.
    """
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS etl_job (
                    year INT PRIMARY KEY,
                    producao_vinhos_mesa DOUBLE PRECISION,
                    exportacao_total_dols DOUBLE PRECISION
                );
                """
            )
        )
        # adiciona coluna de classificação, se não existir
        conn.execute(
            text(
                """
                DO $$
                BEGIN
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


# =========================================================
# Regra ANTIGA de classificação (móvel 5 anos, usando anos anteriores)
# =========================================================
def _classificar_regra_antiga(df_etl: pd.DataFrame) -> pd.DataFrame:
    """
    Usa a regra antiga (média dos últimos 5 anos, olhando APENAS anos anteriores).
    Mapeamento:
      - prod > média & exp > média → 1
      - prod > média & exp < média → 0
      - prod < média & exp > média → 2
      - prod < média & exp < média → 1
    Retorna df com coluna 'desempenho' (int).
    """
    df = df_etl.sort_values("year").reset_index(drop=True).copy()

    # médias móveis usando SOMENTE anos anteriores (shift)
    df["prod_media_5"] = (
        df["producao_vinhos_mesa"].shift(1).rolling(window=5, min_periods=3).mean()
    )
    df["exp_media_5"] = (
        df["exportacao_total_dols"].shift(1).rolling(window=5, min_periods=3).mean()
    )

    # só classifica onde há histórico suficiente
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


# =========================================================
# Job principal: executa TUDO e salva no banco (sem retorno)
# =========================================================
def run_and_save_ml() -> None:
    """
    Pipeline completo:
      - roda ETLs modulares para preencher/atualizar etl_job
      - lê etl_job
      - aplica REGRA ANTIGA (média móvel 5 anos) e grava 'desempenho' em etl_job
      - monta ml_dataset (padronizando nomes de colunas)
      - treina regressão logística e salva métricas
    """
    engine = create_engine(DATABASE_URL, future=True, pool_pre_ping=True)

    # 0) Garante schema/colunas da etl_job
    _ensure_etl_columns(engine)

    # 1) executa ETLs (produção e exportação)
    #    - produção: Vinho de Mesa (litros)
    #    - exportação: Vinhos de mesa / Países: Total (US$)
    run_etl_save()
    run_etl_export_update()

    # 2) lê etl_job
    with engine.begin() as conn:
        etl = pd.read_sql("SELECT * FROM etl_job ORDER BY year", conn)

    # precisa ao menos das 3 colunas
    needed = {"year", "producao_vinhos_mesa", "exportacao_total_dols"}
    if etl.empty or not needed.issubset(set(etl.columns)):
        return

    # 3) classificar com a REGRA ANTIGA
    cls_df = _classificar_regra_antiga(etl)

    # 3.1) atualiza a tabela etl_job com a coluna 'desempenho'
    # mescla pra preservar anos que não têm histórico suficiente (ficam com NULL)
    out_etl = etl.merge(cls_df[["year", "desempenho"]], on="year", how="left")
    with engine.begin() as conn:
        out_etl.to_sql("etl_job", con=conn, if_exists="replace", index=False)
        conn.execute(
            text(
                """
            ALTER TABLE etl_job
            ALTER COLUMN year TYPE INT USING year::INT,
            ALTER COLUMN producao_vinhos_mesa TYPE DOUBLE PRECISION,
            ALTER COLUMN exportacao_total_dols TYPE DOUBLE PRECISION
        """
            )
        )

    # 4) montar ml_dataset (somente anos com rótulo disponível)
    ds = cls_df.rename(
        columns={
            "producao_vinhos_mesa": "producao_total",
            "exportacao_total_dols": "exportacao_total_usd",
        }
    )[["year", "producao_total", "exportacao_total_usd", "desempenho"]]

    if ds.empty or len(ds) < 6:
        _save_ml_dataset(engine, ds)
        return

    _save_ml_dataset(engine, ds)

    # 5) treino e avaliação (com guards)
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
            print("[ML] Abortado: mesmo após novo split, treino continua com 1 classe.")
            return

    # remover multi_class para evitar FutureWarning (sklearn>=1.5)
    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2]).tolist()
    cr = classification_report(y_test, y_pred, labels=[0, 1, 2], output_dict=True)

    _save_ml_metrics(engine, acc, cm, cr, rows=len(ds))
