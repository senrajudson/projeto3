from typing import Optional
from sqlalchemy.orm import Session
import pandas as pd
from db_querys.db_utils import DatasetML, query_scrape_results
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.dialects.postgresql import insert


def extrair_dados_exportacao(
    session: Session,
    tab: str = "Exportação",
    subtab: Optional[str] = "Vinhos de mesa",
    intervalo: tuple[int, int] = (1980, 2023),
) -> pd.DataFrame:
    """Extrai os dados da aba 'Exportação' e subaba 'Vinhos de mesa'."""
    df = query_scrape_results(session, tab=tab, interval=intervalo, subtab=subtab)
    if df.empty:
        print(f"❌ [ETL Exportação] Nenhum dado encontrado para {tab} / {subtab}")
    else:
        print(f"✅ [ETL Exportação] {len(df)} linhas extraídas de {tab} / {subtab}")
    return df


def transformar_exportacao(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra apenas 'Países: Total' e soma 'Valor (US$)' por ano."""
    if df.empty:
        print("❌ [ETL Exportação] DataFrame está vazio após extração.")
        return pd.DataFrame(columns=["ano", "exportacao_total"])

    print("🔍 [ETL Exportação] Colunas disponíveis:", df.columns.tolist())
    print("🔍 [ETL Exportação] Exemplo de dados:\n", df.head())

    # Filtra apenas as linhas com Países == 'Total'
    df_total = df[df["Países"].str.strip().str.lower() == "total"].copy()
    if df_total.empty:
        print("⚠️ [ETL Exportação] Nenhuma linha com Países='Total'.")
        return pd.DataFrame(columns=["ano", "exportacao_total"])

    # Converte 'Valor (US$)' para número
    df_total["Valor (US$)"] = (
        df_total["Valor (US$)"]
        .astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

    # Agrupa por ano (caso haja mais de uma linha por ano)
    df_final = (
        df_total.groupby("ano")["Valor (US$)"]
        .sum()
        .reset_index()
        .rename(columns={"Valor (US$)": "exportacao_total"})
    )

    print("✅ [ETL Exportação] Transformação concluída:\n", df_final.head())
    return df_final


def carregar_em_dataset_ml(session: Session, df_export: pd.DataFrame) -> None:
    """Carrega (insere ou atualiza) os dados de exportação na tabela dataset_ml."""
    if df_export.empty:
        print("⚠️ [ETL Exportação] Nenhum dado para carregar em dataset_ml.")
        return

    try:
        for _, row in df_export.iterrows():
            stmt = (
                insert(DatasetML)
                .values(
                    ano=int(row["ano"]),
                    exportacao_total=str(row["exportacao_total"]),
                )
                .on_conflict_do_update(
                    index_elements=["ano"],
                    set_={"exportacao_total": str(row["exportacao_total"])},
                )
            )
            session.execute(stmt)

        session.commit()
        print(f"✅ [ETL Exportação] {len(df_export)} linhas carregadas em dataset_ml.")
    except SQLAlchemyError as e:
        session.rollback()
        print("❌ [ETL Exportação] Erro ao carregar no banco:", str(e))
        raise
