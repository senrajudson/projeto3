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


class ScrapeRecord(Base):
    __tablename__ = "scrape_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tab = Column(String, nullable=False)
    subtab = Column(String, nullable=True)
    year = Column(Integer, nullable=False)
    data = Column(Text, nullable=False)  # armazena o row como JSON


def init_db(db_filename: str = "scraping.db") -> Session:
    """
    Cria (ou abre) um arquivo SQLite e retorna uma Session ligada a ele.

    - Se estivermos em Vercel/AWS Lambda (detected via VERCEL=1 ou AWS_LAMBDA_FUNCTION_NAME),
      o arquivo será criado em /tmp/<db_filename> (diretório gravável em serverless).
    - Caso contrário (desenvolvimento local), o arquivo será criado ao lado do código:
        ./<db_filename>
    """
    # Detecta se estamos rodando em Vercel (VERCEL=1) ou em Lambda (AWS_LAMBDA_FUNCTION_NAME)
    running_on_serverless = (
        os.environ.get("VERCEL") == "1"
        or os.environ.get("AWS_LAMBDA_FUNCTION_NAME") is not None
    )

    if running_on_serverless:
        # Em Lambda/Vercel, /tmp é o único diretório gravável
        target_dir = Path("/tmp")
        target_dir.mkdir(parents=True, exist_ok=True)  # garante que /tmp exista
        db_path = target_dir / db_filename
        database_url = f"sqlite:///{db_path}"
    else:
        # Em desenvolvimento local, criar o DB na raiz do projeto
        project_root = Path(__file__).parent.parent  # assume que db_utils.py está em db/
        db_path = project_root / db_filename
        db_path.parent.mkdir(parents=True, exist_ok=True)
        database_url = f"sqlite:///{db_path}"

    # Cria engine e session
    engine = create_engine(
        database_url,
        connect_args={"check_same_thread": False},  # necessário para SQLite + threads
        future=True,
    )
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

    # Garante que as tabelas existam
    Base.metadata.create_all(bind=engine)
    return SessionLocal()


def save_scrape_results(
    session: Session, tab: str, df: pd.DataFrame, subtab: Optional[str] = None
) -> None:
    """
    Salva cada linha do DataFrame como um ScrapeRecord no SQLite.

    Parâmetros:
    - session: objeto Session gerado por init_db()
    - tab: nome da aba principal (ex: "Produção")
    - subtab: nome da sub-aba (se houver), caso contrário None
    - df: DataFrame contendo todas as colunas menos 'ano', e com coluna 'ano' indicando o ano
    """
    try:
        records: List[ScrapeRecord] = []
        for _, row in df.iterrows():
            # Converte todas as colunas exceto 'ano' em JSON
            row_dict: Dict[str, Any] = row.drop(labels=["ano"]).to_dict()
            rec = ScrapeRecord(
                tab=tab,
                subtab=subtab,
                year=int(row["ano"]),
                data=json.dumps(row_dict, ensure_ascii=False),
            )
            records.append(rec)

        session.add_all(records)
        session.commit()
    except SQLAlchemyError:
        session.rollback()
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
