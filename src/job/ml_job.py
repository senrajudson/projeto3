# src/ml_job.py
import os
import re
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# =========================================================
# Config
# =========================================================
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@db:5432/projeto3")


# =========================================================
# Helpers de parsing e agregação
# =========================================================
def _coerce_number_br(val: Any) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, (int, float)) and np.isfinite(val):
        return float(val)
    s = str(val).strip()
    s = re.sub(r"[^\d,.\-]", "", s)
    if s.count(",") > 1:
        parts = s.split(",")
        s = "".join(parts[:-1]).replace(".", "") + "." + parts[-1]
    else:
        s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None


def _find_total_value_from_row(
    row_dict: Dict[str, Any], prefer_usd: bool = False
) -> Optional[float]:
    """
    Heurística para pegar 'total' em US$ (exportação) ou não-US$ (produção),
    dependendo de prefer_usd.
    """
    candidates: List[Tuple[str, float]] = []
    for k, v in row_dict.items():
        if v in (None, "", "-"):
            continue
        key_norm = str(k).strip().lower()
        val = _coerce_number_br(v)
        if val is None:
            continue

        if prefer_usd:
            if (
                ("us" in key_norm or "u$" in key_norm)
                and ("total" in key_norm or "valor" in key_norm)
            ) or key_norm in ("us$", "usd"):
                candidates.append((k, val))
        else:
            if "total" in key_norm and ("us" not in key_norm and "u$" not in key_norm):
                candidates.append((k, val))

    if not candidates:
        for k, v in row_dict.items():
            if v in (None, "", "-"):
                continue
            key_norm = str(k).strip().lower()
            val = _coerce_number_br(v)
            if val is None:
                continue
            if "total" in key_norm:
                candidates.append((k, val))

    if not candidates:
        return None
    return max(candidates, key=lambda kv: kv[1])[1]


def _row_is_vinho_de_mesa(row_dict: Dict[str, Any]) -> bool:
    """
    Identifica se a linha representa 'Vinho(s) de Mesa)'.
    Procura em colunas típicas (Produto / Categoria / Tipo) e, como fallback,
    em qualquer valor textual.
    """
    targets = ["vinho de mesa", "vinhos de mesa"]
    # checa colunas típicas
    for key in row_dict.keys():
        k = str(key).strip().lower()
        if k in ("produto", "produto/derivado", "categoria", "tipo"):
            val = str(row_dict.get(key, "")).strip().lower()
            if any(t in val for t in targets):
                return True
    # fallback: procurar em QUALQUER valor textual
    for v in row_dict.values():
        if isinstance(v, str) and any(t in v.strip().lower() for t in targets):
            return True
    return False


def _extract_total_non_usd(row_dict: Dict[str, Any]) -> Optional[float]:
    """
    Extrai um 'total' NÃO US$ (produção em litros/ton etc.) da linha.
    """
    candidates: List[float] = []
    for k, v in row_dict.items():
        if v in (None, "", "-"):
            continue
        key_norm = str(k).strip().lower()
        if "total" in key_norm and ("us" not in key_norm and "u$" not in key_norm):
            val = _coerce_number_br(v)
            if val is not None and np.isfinite(val):
                candidates.append(val)
    if not candidates:
        for k, v in row_dict.items():
            if v in (None, "", "-"):
                continue
            key_norm = str(k).strip().lower()
            if "total" in key_norm:
                val = _coerce_number_br(v)
                if val is not None and np.isfinite(val):
                    candidates.append(val)
    return max(candidates) if candidates else None


def _load_scrapes_df(engine) -> pd.DataFrame:
    """
    Lê scrape_records como DataFrame:
      colunas esperadas: id, tab, subtab, year, data (JSON texto)
    """
    with engine.begin() as conn:
        df = pd.read_sql("SELECT id, tab, subtab, year, data FROM scrape_records", conn)

    # transforma a coluna JSON em dict
    df["data_dict"] = df["data"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else {}
    )
    return df[["tab", "subtab", "year", "data_dict"]]


def _build_yearly_totals(df_scrapes: pd.DataFrame) -> pd.DataFrame:
    """
    Produção (APENAS 'Vinho de Mesa', NÃO total) e Exportação total US$ por ano.
    Heurística:
      - tab começando com 'Produ' → produção
      - tab começando com 'Export' → exportação
    """
    if df_scrapes.empty:
        return pd.DataFrame(columns=["year", "producao_total", "exportacao_total_usd"])

    df = df_scrapes.copy()
    df["tipo"] = np.where(
        df["tab"].str.lower().str.startswith("produ"),
        "producao",
        np.where(
            df["tab"].str.lower().str.startswith("export"), "exportacao", "outros"
        ),
    )

    agg: Dict[int, Dict[str, float]] = {}
    for _, row in df.iterrows():
        ano = int(row["year"])
        tipo = row["tipo"]
        data_dict = row["data_dict"]

        if ano not in agg:
            agg[ano] = {"producao_total": 0.0, "exportacao_total_usd": 0.0}

        if tipo == "producao":
            # SOMENTE "Vinho de Mesa"
            if _row_is_vinho_de_mesa(data_dict):
                v = _extract_total_non_usd(data_dict)
                if v is not None and np.isfinite(v):
                    agg[ano]["producao_total"] += v

        elif tipo == "exportacao":
            # Exportação: total em US$
            v = _find_total_value_from_row(data_dict, prefer_usd=True)
            if v is not None and np.isfinite(v):
                agg[ano]["exportacao_total_usd"] += v

    rows = [{"year": y, **vals} for y, vals in agg.items()]
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def _add_desempenho_labels(df_yearly: pd.DataFrame) -> pd.DataFrame:
    """
    Rótulo 0/1/2 via índice composto (produção e exportação vs média móvel 3 anos)
    e discretização por quantis (tercis). Gera classes mais balanceadas.
    """
    df = df_yearly.sort_values("year").reset_index(drop=True).copy()

    # médias móveis (3 anos ANTERIORES)
    df["prod_media_3"] = (
        df["producao_total"].shift(1).rolling(window=3, min_periods=2).mean()
    )
    df["exp_media_3"] = (
        df["exportacao_total_usd"].shift(1).rolling(window=3, min_periods=2).mean()
    )

    df = df.dropna(subset=["prod_media_3", "exp_media_3"]).copy()

    # desvios relativos (se denom = 0, vira 0)
    def safe_rel(x, m):
        return 0.0 if (m is None or m == 0 or pd.isna(m)) else (x - m) / m

    df["prod_dev"] = [
        safe_rel(p, m) for p, m in zip(df["producao_total"], df["prod_media_3"])
    ]
    df["exp_dev"] = [
        safe_rel(e, m) for e, m in zip(df["exportacao_total_usd"], df["exp_media_3"])
    ]

    # índice composto (ajuste pesos se quiser)
    df["score"] = 0.5 * df["prod_dev"] + 0.5 * df["exp_dev"]

    # para evitar erro do qcut quando há muitos empates, use rank com jitter leve
    s = df["score"].astype(float)
    # se tudo igual, o rank protege:
    if s.nunique(dropna=True) < 3:
        s = s.rank(method="first")  # 1..N

    # 3 faixas por quantis
    try:
        df["desempenho"] = pd.qcut(s, q=3, labels=[0, 1, 2]).astype(int)
    except ValueError:
        # fallback se qcut não conseguir por empates extremos: use cut em faixas iguais
        df["desempenho"] = pd.cut(
            s, bins=3, labels=[0, 1, 2], include_lowest=True
        ).astype(int)

    return df[["year", "producao_total", "exportacao_total_usd", "desempenho"]]


# =========================================================
# Persistência
# =========================================================
def _ensure_metrics_table(engine):
    """
    Cria tabela ml_metrics se não existir (acurácia, cm, report JSON, rows, created_at).
    """
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
    """
    Salva (replace) o dataset tratado em 'ml_dataset'.
    """
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


# =========================================================
# Job principal: executa TUDO e salva no banco (sem retorno)
# =========================================================
def run_and_save_ml() -> None:
    """
    Executa o pipeline completo e registra resultados no banco:
      - lê scrape_records
      - agrega por ano (produção = apenas 'Vinho de Mesa')
      - calcula desempenho (média móvel dos 3 anos anteriores)
      - salva ml_dataset
      - treina Regressão Logística
      - salva métricas em ml_metrics
    Não retorna nada (pensado para ser chamado pela API).
    """
    engine = create_engine(DATABASE_URL, future=True)

    # 1) carregar scrapes
    df_scrapes = _load_scrapes_df(engine)
    if df_scrapes.empty:
        return

    # 2) totais anuais
    yearly = _build_yearly_totals(df_scrapes)
    if yearly.empty:
        return

    # 3) class labels
    ds = _add_desempenho_labels(yearly)
    if (
        ds.empty or len(ds) < 4
    ):  # janela 3 => normalmente dá para trabalhar com >=4 anos após o corte
        _save_ml_dataset(engine, ds)
        return

    # 4) salvar dataset tratado
    _save_ml_dataset(engine, ds)

    # 5) treino e avaliação (com guards)
    X = ds[["year", "producao_total", "exportacao_total_usd"]].values
    y = ds["desempenho"].values

    classes, counts = np.unique(y, return_counts=True)
    if classes.size < 2:
        # salva dataset mesmo assim e encerra sem erro
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
        # tenta split alternativo
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.8, random_state=42
        )
        if np.unique(y_train).size < 2:
            print("[ML] Abortado: mesmo após novo split, treino continua com 1 classe.")
            return

    model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 6) métricas → salvar em ml_metrics
    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2]).tolist()
    cr = classification_report(y_test, y_pred, labels=[0, 1, 2], output_dict=True)

    _save_ml_metrics(engine, acc, cm, cr, rows=len(ds))
