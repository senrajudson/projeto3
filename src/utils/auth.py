# src/utils/auth.py

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

# === Camada de dados (sempre Postgres/SQLAlchemy) ===
from db_querys.db_users import (
    get_db as get_db_session,  # -> Generator[Session, None, None]
    get_user_db as get_user_from_db,  # (db, username) -> UserModel
    create_user_db as create_user_in_db,  # (db, username, password, full_name) -> UserModel
    authenticate_user_db as authenticate_user_in_db,  # (db, username, password) -> UserModel|None
    User as UserModelDB,  # SQLAlchemy model
)

# === Configurações de JWT ===
SECRET_KEY = os.getenv("SECRET_KEY", "CHANGE_ME_IN_PRODUCTION")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
router = APIRouter()


# === Schemas Pydantic (respostas da API) ===
class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


class User(BaseModel):
    username: str
    full_name: Optional[str] = None
    disabled: Optional[bool] = False


class UserCreate(BaseModel):
    username: str
    password: str
    full_name: Optional[str] = None


# === Helpers JWT ===
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


# === Dependencies ===
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
        username: Optional[str] = payload.get("sub")
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
    if getattr(current_user, "disabled", False):
        raise HTTPException(status_code=400, detail="Usuário inativo")
    return current_user


# === Rotas de Auth ===
@router.post(
    "/users", response_model=User, status_code=201, summary="Cria novo usuário"
)
def create_user(
    user_in: UserCreate,
    db_session=Depends(get_db_session),
):
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
        disabled=getattr(user_obj, "disabled", False),
    )


@router.post("/token", response_model=Token, summary="Gera token JWT de acesso")
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db_session=Depends(get_db_session),
):
    user = authenticate_user_in_db(db_session, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Usuário ou senha incorretos",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(data={"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}
