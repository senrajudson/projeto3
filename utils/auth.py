from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import (
    OAuth2PasswordBearer,
    OAuth2PasswordRequestForm,
)
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from db.db_users import (
    get_db,
    get_user_db,
    create_user_db,
    authenticate_user_db,
    User as UserModel,
    get_db as get_user_db_session,
)

# --- Configurações de JWT ---
SECRET_KEY = "YOUR_SECRET_KEY"       # troque em produção!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

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
    db = Depends(get_user_db_session),
) -> UserModel:
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
    user = get_user_db(db, username)
    if not user:
        raise credentials_exc
    return user


def get_current_active_user(
    current_user: UserModel = Depends(get_current_user),
) -> UserModel:
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Usuário inativo")
    return current_user


@router.post("/users", response_model=User, status_code=201, summary="Cria novo usuário")
def create_user(
    user_in: UserCreate,
    db = Depends(get_user_db_session),
):
    """
    Cria um novo usuário com senha.
    """
    existing = get_user_db(db, user_in.username)
    if existing:
        raise HTTPException(status_code=400, detail="Usuário já existe")
    user = create_user_db(
        db,
        username=user_in.username,
        password=user_in.password,
        full_name=user_in.full_name,
    )
    return User(
        username=user.username,
        full_name=user.full_name,
        disabled=user.disabled,
    )


@router.post("/token", response_model=Token, summary="Gera token JWT de acesso")
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db = Depends(get_user_db_session),
):
    """
    Autentica usuário e retorna token JWT.
    """
    user = authenticate_user_db(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Usuário ou senha incorretos",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(data={"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}
