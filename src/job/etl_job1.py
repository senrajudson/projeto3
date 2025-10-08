# src/job/etl_job.py
import os
import re
import json
from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

# Lê do ambiente; pode ser sobrescrito via argumento da função
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@db:5432/projeto3")


# ---------- Parsers & heurísticas ----------
def _coerce_number_br(val: Any) -> Optional[float]:
    """Converte '102.631.280' ou '1.234,56' para float (litros)."""
    if val is None:
        return None
    if isinstance(val, (int, float)) and np.isfinite(val):
        return float(val)
    s = str(val).strip()
    if not s:
        return None
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


def _row_is_vinho_de_mesa(row_dict: Dict[str, Any]) -> bool:
    """Detecta 'Vinho(s) de Mesa' (case-insensitive) em campos comuns ou em qualquer valor textual."""
    targets = ("vinho de mesa", "vinhos de mesa")
    for key in ("Produto", "Produto/Derivado", "Categoria", "Tipo"):
        if key in row_dict and isinstance(row_dict[key], str):
            v = row_dict[key].strip().lower()
            if any(t in v for t in targets):
                return True
    for v in row_dict.values():
        if isinstance(v, str) and any(t in v.strip().lower() for t in targets):
            return True
    return False


def _extract_quantidade_litros(row_dict: Dict[str, Any]) -> Optional[float]:
    """
    Extrai quantidade em litros.
    Prioriza chaves com 'quantidade' + referência a litro; senão, qualquer 'quantidade';
    último recurso: maior número da linha.
    """
    for k, v in row_dict.items():
        if not isinstance(k, str):
            continue
        lk = k.lower()
        if "quantidade" in lk and (
            "(l" in lk or "litro" in lk or "litros" in lk or " l" in lk
        ):
            num = _coerce_number_br(v)
            if num is not None:
                return num

    for k, v in row_dict.items():
        if not isinstance(k, str):
            continue
        lk = k.lower()
        if "quantidade" in lk:
            num = _coerce_number_br(v)
            if num is not None:
                return num

    numbers: List[float] = []
    for v in row_dict.values():
        num = _coerce_number_br(v)
        if num is not None:
            numbers.append(num)
    return max(numbers) if numbers else None


def _create_target_table_if_needed(engine) -> None:
    ddl = """
    CREATE TABLE IF NOT EXISTS etl_job (
        year INT PRIMARY KEY,
        producao_vinhos_mesa DOUBLE PRECISION NOT NULL
    );
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


def _truncate_target(engine) -> None:
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE etl_job;"))


# ---------- Função pública (modular) ----------
def run_etl_save(
    database_url: Optional[str] = None, return_df: bool = False
) -> Optional[pd.DataFrame]:
    """
    Executa o ETL e grava em public.etl_job (colunas: year, producao_vinhos_mesa em litros).
    - Lê public.scrape_records (tab, subtab, year, data JSON)
    - Filtra Produção + 'Vinho(s) de Mesa'
    - Extrai 'Quantidade (L.)' (ou equivalente) e agrega por ano
    - Substitui o conteúdo da tabela etl_job (idempotente)

    Args:
        database_url: URL do Postgres; se None, usa env DATABASE_URL.
        return_df: se True, retorna o DataFrame agregado.

    Returns:
        DataFrame agregado (year, producao_vinhos_mesa) se return_df=True; caso contrário, None.
    """
    db_url = database_url or DATABASE_URL
    engine = create_engine(db_url, future=True, pool_pre_ping=True)

    with engine.begin() as conn:
        df = pd.read_sql("SELECT tab, subtab, year, data FROM scrape_records", conn)

    # Garantes a tabela destino existir
    _create_target_table_if_needed(engine)

    if df.empty:
        _truncate_target(engine)
        return (
            pd.DataFrame(columns=["year", "producao_vinhos_mesa"])
            if return_df
            else None
        )

    df["data_dict"] = df["data"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else {}
    )
    df["is_producao"] = df["tab"].str.lower().str.startswith("produ")
    mask = df["is_producao"] & df["data_dict"].apply(_row_is_vinho_de_mesa)
    df_vm = df[mask].copy()

    if df_vm.empty:
        _truncate_target(engine)
        return (
            pd.DataFrame(columns=["year", "producao_vinhos_mesa"])
            if return_df
            else None
        )

    df_vm["qtd_litros"] = df_vm["data_dict"].apply(_extract_quantidade_litros)
    df_vm = df_vm.dropna(subset=["qtd_litros"])

    if df_vm.empty:
        _truncate_target(engine)
        return (
            pd.DataFrame(columns=["year", "producao_vinhos_mesa"])
            if return_df
            else None
        )

    agg = (
        df_vm.groupby("year", as_index=False)["qtd_litros"]
        .sum()
        .rename(columns={"qtd_litros": "producao_vinhos_mesa"})
        .sort_values("year")
        .reset_index(drop=True)
    )

    with engine.begin() as conn:
        agg.to_sql("etl_job", con=conn, if_exists="replace", index=False)
        conn.execute(
            text(
                """
            ALTER TABLE etl_job
            ALTER COLUMN year TYPE INT USING year::INT,
            ALTER COLUMN producao_vinhos_mesa TYPE DOUBLE PRECISION;
        """
            )
        )

    return agg if return_df else None
