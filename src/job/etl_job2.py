import os
import re
import json
from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@db:5432/projeto3")


# ----------------- helpers -----------------
def _coerce_number_br(val: Any) -> Optional[float]:
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


def _is_export_vinhos_de_mesa(tab: str, subtab: Optional[str]) -> bool:
    t = (tab or "").strip().lower()
    s = (subtab or "").strip().lower()
    return t.startswith("export") and ("vinhos de mesa" in s or s == "")


def _is_paises_total(row: Dict[str, Any]) -> bool:
    for k, v in row.items():
        if isinstance(k, str) and k.strip().lower().startswith("países"):
            return isinstance(v, str) and v.strip().lower() == "total"
    for v in row.values():
        if isinstance(v, str) and v.strip().lower() == "total":
            return True
    return False


def _extract_valor_usd(row: Dict[str, Any]) -> Optional[float]:
    for k, v in row.items():
        if not isinstance(k, str):
            continue
        lk = k.strip().lower()
        if "valor" in lk and ("us$" in lk or "usd" in lk or "us" in lk):
            num = _coerce_number_br(v)
            if num is not None:
                return num
    candidates: List[float] = []
    for k, v in row.items():
        if isinstance(k, str) and ("us" in k.lower() or "u$" in k.lower()):
            num = _coerce_number_br(v)
            if num is not None:
                candidates.append(num)
    return max(candidates) if candidates else None


def _ensure_target_table(engine) -> None:
    ddl = """
    CREATE TABLE IF NOT EXISTS etl_job (
        year INT PRIMARY KEY,
        producao_vinhos_mesa DOUBLE PRECISION
    );
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))
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
            END$$;
        """
            )
        )


def _extract_export_usd_by_year(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Filtra Exportação/Vinhos de mesa, Países=Total e agrega Valor (US$) por ano."""
    if df_raw.empty:
        return pd.DataFrame(columns=["year", "exportacao_total_dols"])

    df = df_raw.copy()
    df["ok"] = df.apply(
        lambda r: _is_export_vinhos_de_mesa(r["tab"], r["subtab"]), axis=1
    )
    df = df[df["ok"]].copy()
    if df.empty:
        return pd.DataFrame(columns=["year", "exportacao_total_dols"])

    df["data_dict"] = df["data"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else {}
    )
    df["is_total"] = df["data_dict"].apply(_is_paises_total)
    df = df[df["is_total"]].copy()
    if df.empty:
        return pd.DataFrame(columns=["year", "exportacao_total_dols"])

    df["valor_usd"] = df["data_dict"].apply(_extract_valor_usd)
    df = df.dropna(subset=["valor_usd"])
    if df.empty:
        return pd.DataFrame(columns=["year", "exportacao_total_dols"])

    out = (
        df.groupby("year", as_index=False)["valor_usd"]
        .sum()
        .rename(columns={"valor_usd": "exportacao_total_dols"})
        .sort_values("year")
        .reset_index(drop=True)
    )
    return out


def run_etl_export_update(
    database_url: Optional[str] = None, return_df: bool = False
) -> Optional[pd.DataFrame]:
    """
    Atualiza a coluna exportacao_total_dols em etl_job **apenas** para anos já existentes,
    sem usar to_sql(..., replace). Evita perder colunas.
    """
    db_url = database_url or DATABASE_URL
    engine = create_engine(db_url, future=True, pool_pre_ping=True)

    # Garante tabela e coluna alvo
    _ensure_target_table(engine)

    # Lê anos existentes na etl_job
    with engine.begin() as conn:
        etl = pd.read_sql("SELECT * FROM etl_job", conn)

    if etl.empty or "year" not in etl.columns:
        return etl if return_df else None

    years = sorted(etl["year"].dropna().astype(int).unique().tolist())

    # Busca scrapes apenas desses anos
    with engine.begin() as conn:
        raw = pd.read_sql(
            text(
                "SELECT tab, subtab, year, data FROM scrape_records WHERE year = ANY(:years)"
            ),
            conn,
            params={"years": years},
        )

    # Extrai valores US$ por ano
    usd_by_year = _extract_export_usd_by_year(raw)
    if usd_by_year.empty:
        return etl if return_df else None

    # UPDATE por ano (sem derrubar a tabela)
    with engine.begin() as conn:
        for _, r in usd_by_year.iterrows():
            conn.execute(
                text("UPDATE etl_job SET exportacao_total_dols=:usd WHERE year=:year"),
                {"year": int(r["year"]), "usd": float(r["exportacao_total_dols"])},
            )

    if not return_df:
        return None

    with engine.begin() as conn:
        final_df = pd.read_sql("SELECT * FROM etl_job ORDER BY year", conn)
    return final_df


# ----------------- Função 2: UPSERT para todos os anos encontrados -----------------
def run_etl_export_upsert_all(
    database_url: Optional[str] = None, return_df: bool = False
) -> Optional[pd.DataFrame]:
    """
    Varre todos os anos existentes em scrape_records (Exportação/Vinhos de mesa/Países: Total),
    extrai Valor(US$) e faz UPSERT em etl_job (cria ano se não existir).
    """
    db_url = database_url or DATABASE_URL
    engine = create_engine(db_url, future=True, pool_pre_ping=True)
    _ensure_target_table(engine)

    # pega tudo uma vez (pode filtrar por intervalo se quiser)
    with engine.begin() as conn:
        raw = pd.read_sql("SELECT tab, subtab, year, data FROM scrape_records", conn)

    usd_by_year = _extract_export_usd_by_year(raw)
    if usd_by_year.empty:
        with engine.begin() as conn:
            etl_now = pd.read_sql("SELECT * FROM etl_job", conn)
        return etl_now if return_df else None

    # UPSERT linha-a-linha (simples e claro)
    upsert_sql = text(
        """
        INSERT INTO etl_job (year, exportacao_total_dols)
        VALUES (:year, :usd)
        ON CONFLICT (year)
        DO UPDATE SET exportacao_total_dols = EXCLUDED.exportacao_total_dols;
    """
    )
    with engine.begin() as conn:
        for _, r in usd_by_year.iterrows():
            conn.execute(
                upsert_sql,
                {"year": int(r["year"]), "usd": float(r["exportacao_total_dols"])},
            )
        # padroniza tipos e mantém outras colunas
        conn.execute(
            text(
                """
            ALTER TABLE etl_job
            ALTER COLUMN year TYPE INT USING year::INT,
            ALTER COLUMN producao_vinhos_mesa TYPE DOUBLE PRECISION,
            ALTER COLUMN exportacao_total_dols TYPE DOUBLE PRECISION;
        """
            )
        )

    if not return_df:
        return None

    with engine.begin() as conn:
        final_df = pd.read_sql("SELECT * FROM etl_job ORDER BY year", conn)
    return final_df
