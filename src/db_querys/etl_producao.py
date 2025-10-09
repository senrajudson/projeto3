# db_querys/etl_producao.py

from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert
import pandas as pd
import re

from db_querys.db_utils import DatasetML, query_scrape_results, init_db


def coerce_number_br(val: str) -> float:
    """
    Converte números brasileiros do tipo '102.631.280' em float.
    """
    val = val.strip()
    val = val.replace(".", "").replace(",", ".")
    return float(val)


def extrair_dados_producao(
    session: Session, intervalo: tuple[int, int]
) -> pd.DataFrame:
    df = query_scrape_results(session, tab="Produção", interval=intervalo)
    return df


def transformar_producao(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["ano", "producao_litros"])

    produtos_alvo = ["VINHO DE MESA", "VINHO FINO DE MESA (VINIFERA)"]

    df_filtrado = df[df["Produto"].isin(produtos_alvo)].copy()

    df_filtrado["Quantidade (L.)"] = df_filtrado["Quantidade (L.)"].apply(
        coerce_number_br
    )

    df_agrupado = (
        df_filtrado.groupby("ano")["Quantidade (L.)"]
        .sum()
        .reset_index()
        .rename(columns={"Quantidade (L.)": "producao_litros"})
    )

    return df_agrupado


def carregar_em_dataset_ml(session: Session, df_producao: pd.DataFrame) -> None:
    try:
        for _, row in df_producao.iterrows():
            stmt = (
                insert(DatasetML)
                .values(
                    ano=int(row["ano"]),
                    producao_litros=str(row["producao_litros"]),
                )
                .on_conflict_do_update(
                    index_elements=["ano"],
                    set_={"producao_litros": str(row["producao_litros"])},
                )
            )
            session.execute(stmt)

        session.commit()
    except Exception:
        session.rollback()
        raise
