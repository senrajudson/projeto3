from typing import Optional
from sqlalchemy.orm import Session
import pandas as pd

from db_querys.db_utils import DatasetML, query_scrape_results, init_db


def extrair_dados_exportacao(
    session: Session, tab: str, intervalo: tuple[int, int]
) -> pd.DataFrame:
    df = query_scrape_results(session, tab=tab, interval=intervalo)
    return df


def transformar_exportacao(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["ano", "exportacao_total"])

    df_filtrado = df[df["Países"] == "Total"].copy()
    df_resultado = df_filtrado[["ano", "Valor (US$)"]].rename(
        columns={"Valor (US$)": "exportacao_total"}
    )
    return df_resultado


def carregar_em_dataset_ml(session: Session, df_export: pd.DataFrame) -> None:
    from sqlalchemy.exc import SQLAlchemyError
    from sqlalchemy.dialects.postgresql import insert

    try:
        for _, row in df_export.iterrows():
            stmt = (
                insert(DatasetML)
                .values(
                    ano=int(row["ano"]), exportacao_total=str(row["exportacao_total"])
                )
                .on_conflict_do_update(
                    index_elements=["ano"],
                    set_={"exportacao_total": str(row["exportacao_total"])},
                )
            )
            session.execute(stmt)

        session.commit()
    except SQLAlchemyError:
        session.rollback()
        raise
