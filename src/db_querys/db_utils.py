import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import Column, Integer, String, Text, create_engine, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

Base = declarative_base()


class DatasetML(Base):
    __tablename__ = "dataset_ml"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ano = Column(Integer, nullable=False, unique=True)
    exportacao_total = Column(String, nullable=True)
    producao_litros = Column(String, nullable=True)  # <-- ADICIONE ESTA LINHA


class ScrapeRecord(Base):
    __tablename__ = "scrape_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tab = Column(String, nullable=False)
    subtab = Column(String, nullable=True)
    year = Column(Integer, nullable=False)
    data = Column(Text, nullable=False)  # armazena o row como JSON


def init_db(database_url: Optional[str] = None) -> Session:
    """
    Inicializa a conexão com o banco PostgreSQL (ou outro) via SQLAlchemy.
    Retorna uma session.
    """

    # Tenta obter a URL do banco de variável de ambiente ou argumento
    if database_url is None:
        database_url = os.getenv(
            "DATABASE_URL", "postgresql://user:password@db:5432/projeto3"
        )

    engine = create_engine(database_url, future=True)
    SessionLocal = sessionmaker(
        bind=engine, autoflush=False, autocommit=False, future=True
    )

    # Cria as tabelas, se não existirem
    Base.metadata.create_all(bind=engine)
    return SessionLocal()


def save_scrape_results(
    session: Session, tab: str, df: pd.DataFrame, subtab: Optional[str] = None
) -> None:
    """
    Salva cada linha do DataFrame como um ScrapeRecord no banco.

    Parâmetros:
    - session: objeto Session gerado por init_db()
    - tab: nome da aba principal (ex: "Produção")
    - subtab: nome da sub-aba (se houver), caso contrário None
    - df: DataFrame contendo todas as colunas menos 'ano', e com coluna 'ano' indicando o ano
    """
    from sqlalchemy.exc import SQLAlchemyError

    try:
        records: List[ScrapeRecord] = []
        for _, row in df.iterrows():
            row = row.dropna(how="all")  # ignora linhas completamente vazias

            # Verifica se 'ano' existe e é válido
            if "ano" not in row or pd.isna(row["ano"]):
                continue

            # Extrai as colunas, removendo 'ano'
            row_dict: Dict[str, Any] = row.drop(labels=["ano"]).to_dict()

            # Cria o objeto ScrapeRecord
            rec = ScrapeRecord(
                tab=tab,
                subtab=subtab,
                year=int(row["ano"]),
                data=json.dumps(row_dict, ensure_ascii=False),
            )
            records.append(rec)

        # Salva no banco
        session.add_all(records)
        session.commit()

        # Log opcional: anos salvos
        anos_salvos = sorted(set([r.year for r in records]))
        print(
            f"✅ [save_scrape_results] {len(records)} registros salvos para anos: {anos_salvos}"
        )

    except SQLAlchemyError as e:
        session.rollback()
        print(f"❌ Erro ao salvar registros no banco: {e}")
        raise


def query_scrape_results(
    session: Session, tab: str, interval: Tuple[int, int], subtab: Optional[str] = None
) -> pd.DataFrame:
    """
    Busca no banco todos os registros que pertençam à aba 'tab',
    sub-aba 'subtab' (se fornecida) e cujo ano esteja no intervalo [start, end].

    Retorna um DataFrame reconstruído, onde cada linha tem as colunas originais
    (extraídas do campo JSON 'data') e também a coluna 'ano'.
    """
    start, end = sorted(interval)
    stmt = (
        select(ScrapeRecord)
        .where(ScrapeRecord.tab == tab)
        .where(ScrapeRecord.year.between(start, end))
    )
    if subtab is not None:
        stmt = stmt.where(ScrapeRecord.subtab == subtab)

    results = session.execute(stmt).scalars().all()

    rows: List[Dict[str, Any]] = []
    for rec in results:
        data = json.loads(rec.data)
        data["ano"] = rec.year
        if rec.subtab:
            data["subtab"] = rec.subtab
        rows.append(data)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)
