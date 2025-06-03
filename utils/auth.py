# utils/auth.py

import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

# Se estivermos rodando localmente (ou em qualquer outro lugar que não seja Vercel), 
# usaremos SQLAlchemy para persistir usuários em SQLite.
# Caso contrário (VERCEL="1"), usaremos um simples dict em memória.
RUNNING_ON_VERCEL = os.environ.get("VERCEL") == "1"

if not RUNNING_ON_VERCEL:
    # Importa a camada de DB real (SQLAlchemy + SQLite)
    from db.db_users import (
        get_db as _get_db_session,
        get_user_db as _get_user_db,
        create_user_db as _create_user_db,
        authenticate_user_db as _authenticate_user_db,
        User as UserModelDB,
    )
else:
    # Preparar um dicionário simples em memória para simular usuários.
    # A chave do dicionário será o username, e o valor conterá os campos do UserInDB.
    from typing import List, Tuple

    # Emulação de tabela de usuários em memória
    FAKE_USERS_DB: Dict[str, Dict[str, Any]] = {}

    class UserModelDB(BaseModel):
        username: str
        full_name: Optional[str] = None
        disabled: bool = False
        hashed_password: str

    def _get_db_session():
        """
        Placeholder para manter a assinatura igual ao get_db real,
        mas não faz nada. Só para satisfazer a dependência do FastAPI.
        """
        return None

    def _get_user_db(db_session, username: str) -> Optional[UserModelDB]:
        """
        Busca usuário no dicionário FAKE_USERS_DB.
        """
        data = FAKE_USERS_DB.get(username)
        if not data:
            return None
        return UserModelDB(**data)

    def _create_user_db(db_session, username: str, password: str, full_name: Optional[str] = None):
        """
        Cria um usuário no dicionário em memória. Retorna um objeto UserModelDB
        equivalente àquele retornado pelo SQLAlchemy.
        """
        hashed_password = pwd_context.hash(password)
        user_dict = {
            "username": username,
            "full_name": full_name,
            "disabled": False,
            "hashed_password": hashed_password,
        }
        FAKE_USERS_DB[username] = user_dict
        return UserModelDB(**user_dict)

    def _authenticate_user_db(db_session, username: str, password: str):
        """
        Verifica se o username existe em FAKE_USERS_DB e se a senha bate.
        Retorna o UserModelDB se ok, caso contrário retorna False.
        """
        user = _get_user_db(None, username)
        if not user:
            return False
        if not pwd_context.verify(password, user.hashed_password):
            return False
        return user


# ----------------------------------------------------------------------
# Agora, independentemente de estarmos no Vercel ou não, 
# as variáveis a seguir apontam para a função “certa”.
get_db_session = _get_db_session          # assinatura: get_db_session() -> Session ou None
get_user_from_db = _get_user_db           # assinatura: get_user_from_db(db_session, username)
create_user_in_db = _create_user_db       # assinatura: create_user_in_db(db_session, username, password, full_name)
authenticate_user_in_db = _authenticate_user_db  # assinatura: authenticate_user_in_db(db_session, username, password)
# ----------------------------------------------------------------------

# --- Configurações de JWT ---
SECRET_KEY = "YOUR_SECRET_KEY"       # Em produção, troque por algo bem mais seguro!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

router = APIRouter()


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


class User(BaseModel):
    username: str
    full_name: Optional[str] = None
    disabled: Optional[bool] = False


class UserInDB(User):
    hashed_password: str


class UserCreate(BaseModel):
    username: str
    password: str
    full_name: Optional[str] = None


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db_session=Depends(get_db_session),
) -> UserModelDB:
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Não foi possível validar credenciais",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if not username:
            raise credentials_exc
    except JWTError:
        raise credentials_exc

    user = get_user_from_db(db_session, username)
    if not user:
        raise credentials_exc
    return user


def get_current_active_user(
    current_user: UserModelDB = Depends(get_current_user),
) -> UserModelDB:
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Usuário inativo")
    return current_user


@router.post("/users", response_model=User, status_code=201, summary="Cria novo usuário")
def create_user(
    user_in: UserCreate,
    db_session=Depends(get_db_session),
):
    """
    Cria um novo usuário com senha. Em Vercel, armazena em memória; 
    caso contrário, grava no SQLite.
    """
    existing = get_user_from_db(db_session, user_in.username)
    if existing:
        raise HTTPException(status_code=400, detail="Usuário já existe")

    user_obj = create_user_in_db(
        db_session,
        username=user_in.username,
        password=user_in.password,
        full_name=user_in.full_name,
    )
    return User(
        username=user_obj.username,
        full_name=user_obj.full_name,
        disabled=user_obj.disabled,
    )


@router.post("/token", response_model=Token, summary="Gera token JWT de acesso")
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db_session=Depends(get_db_session),
):
    """
    Autentica usuário e retorna token JWT.
    Se Vercel, busca no dicionário em memória; caso contrário, faz query no SQLite.
    """
    user = authenticate_user_in_db(db_session, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Usuário ou senha incorretos",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(data={"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}
