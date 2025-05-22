import json
# from pathlib import Path
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


def init_db(db_path: str = "scraping.db") -> Session:
    """
    Cria o arquivo .db (se não existir) e retorna uma Session ligada a ele.
    """
    # Ex.: sqlite:///scraping.db
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, future=True)()


def save_scrape_results(
    session: Session,
    tab: str,
    df: pd.DataFrame,
    subtab: Optional[str] = None
) -> None:
    """
    Salva cada linha do DataFrame como um ScrapeRecord.  
    - `tab`: nome da aba principal  
    - `subtab`: nome da sub-aba (se houver)  
    """
    try:
        records = []
        for _, row in df.iterrows():
            # converte a linha inteira para dict, menos a coluna 'ano'
            row_dict: Dict[str, Any] = row.drop(labels=["ano"]).to_dict()
            rec = ScrapeRecord(
                tab=tab,
                subtab=subtab,
                year=int(row["ano"]),
                data=json.dumps(row_dict, ensure_ascii=False)
            )
            records.append(rec)
        session.add_all(records)
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        raise


def query_scrape_results(
    session: Session,
    tab: str,
    interval: Tuple[int, int],
    subtab: Optional[str] = None
) -> pd.DataFrame:
    """
    Consulta todos os ScrapeRecord que casem com aba, sub-aba (se fornecida)
    e ano dentro do intervalo, retornando num DataFrame.
    """
    start, end = sorted(interval)
    stmt = select(ScrapeRecord).where(
        ScrapeRecord.tab == tab,
        ScrapeRecord.year.between(start, end)
    )
    if subtab is not None:
        stmt = stmt.where(ScrapeRecord.subtab == subtab)

    results = session.execute(stmt).scalars().all()

    # reconstrói DataFrame expandindo o JSON de cada record
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

