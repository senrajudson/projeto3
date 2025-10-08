from typing import Optional, Generator
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Boolean,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from passlib.context import CryptContext

# 1) Configurações do banco PostgreSQL (usa variáveis fixas ou os.getenv)
DATABASE_URL = "postgresql://user:password@db:5432/projeto3"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# 2) Contexto para hash de senhas
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# 3) Modelo
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    disabled = Column(Boolean, default=False)


# 4) Cria a tabela (execute uma vez)
Base.metadata.create_all(bind=engine)


# 5) Dependência para injetar a sessão no FastAPI
def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# 6) Operações CRUD
def get_user_db(db: Session, username: str) -> Optional[User]:
    return db.query(User).filter(User.username == username).first()


def create_user_db(
    db: Session,
    username: str,
    password: str,
    full_name: Optional[str] = None,
    disabled: bool = False,
) -> User:
    hashed = pwd_context.hash(password)
    user = User(
        username=username,
        hashed_password=hashed,
        full_name=full_name,
        disabled=disabled,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user_db(db: Session, username: str, password: str) -> Optional[User]:
    user = get_user_db(db, username)
    if not user or not pwd_context.verify(password, user.hashed_password):
        return None
    return user
