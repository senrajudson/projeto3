from fastapi import FastAPI, Query, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from contextlib import asynccontextmanager
from typing import List, Tuple, Sequence, Optional
import subprocess
import sys
from datetime import datetime, timedelta

import pandas as pd
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError
from pydantic import BaseModel
from passlib.context import CryptContext
from jose import JWTError, jwt

from db.db_utils import init_db, save_scrape_results, query_scrape_results, ScrapeRecord
from db.db_users import (
    get_db,
    get_user_db,
    create_user_db,
    authenticate_user_db,
    User as UserModel,
)


# JWT config
SECRET_KEY = "YOUR_SECRET_KEY"  # troque em produção!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan handler: garante que o Playwright Chromium esteja instalado no startup.
    """
    try:
        subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as e:
        print("❌ Falha ao instalar o Chromium:", e)
        sys.exit(1)
    yield


app = FastAPI(
    title="Embrapa Scraper API",
    description="Scraping das abas do site da Embrapa com cache em SQLite e JWT auth",
    version="1.0.0",
    lifespan=lifespan,
)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db=Depends(get_db)
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
    current_user: UserModel = Depends(get_current_user)
) -> UserModel:
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Usuário inativo")
    return current_user


@app.post("/users", response_model=User, status_code=201, summary="Cria novo usuário")
def create_user(
    user_in: UserCreate,
    db=Depends(get_db)
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
        full_name=user_in.full_name
    )
    return User(username=user.username, full_name=user.full_name, disabled=user.disabled)


@app.post("/token", response_model=Token, summary="Gera token JWT de acesso")
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db=Depends(get_db)
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


async def _scrape_year(page, year: int) -> pd.DataFrame:
    """
    Extrai os dados da tabela para um ano específico.
    """
    await page.fill("input.text_pesq", str(year), timeout=2000)
    await page.press("input.text_pesq", "Enter")

    try:
        await page.wait_for_selector("table.tb_base.tb_dados", timeout=5000)
    except TimeoutError:
        return pd.DataFrame()

    soup = BeautifulSoup(await page.content(), "lxml")
    table = soup.select_one("table.tb_base.tb_dados")
    if not table:
        return pd.DataFrame()

    headers = [th.get_text(strip=True) for th in table.select("thead th")]

    rows = []
    for tr in table.select("tbody tr"):
        cols = [td.get_text(strip=True) for td in tr.select("td")]
        if len(cols) == len(headers):
            rows.append(cols)

    total = [td.get_text(strip=True) for td in table.select("tfoot td")]
    if len(total) == len(headers):
        rows.append(total)

    df = pd.DataFrame(rows, columns=headers)
    df["ano"] = year
    return df


async def fetch_scrape(
    interval: Tuple[int, int],
    tabs: Sequence[str]
) -> pd.DataFrame:
    """
    Executa o scraping para cada ano no intervalo e concatena os resultados.
    """
    start_year, end_year = sorted(interval)
    years = range(start_year, end_year + 1)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        page.set_default_timeout(5000)
        page.set_default_navigation_timeout(5000)

        await page.goto("http://vitibrasil.cnpuv.embrapa.br/", wait_until="networkidle")

        try:
            await page.click('button:has-text("Aceito")', timeout=2000)
        except TimeoutError:
            pass

        for label in tabs:
            try:
                await page.click(f'button:has-text("{label}")', timeout=2000)
            except TimeoutError:
                raise HTTPException(status_code=404,
                                    detail=f"Aba/subaba '{label}' não encontrada")

        dfs = []
        for year in years:
            df_year = await _scrape_year(page, year)
            if not df_year.empty:
                dfs.append(df_year)

        await browser.close()

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def get_cached_years_db(
    session,
    main_tab: str,
    subtab: Optional[str]
) -> set[int]:
    """
    Retorna o conjunto de anos já persistidos no banco para a combinação
    de aba e subaba informadas.
    """
    query = session.query(ScrapeRecord.year).filter(
        ScrapeRecord.tab == main_tab
    )
    if subtab:
        query = query.filter(ScrapeRecord.subtab == subtab)
    results = query.distinct().all()
    return {r[0] for r in results}


@app.get(
    "/scrape",
    summary="Realiza scraping com cache e retorna JSON",
    dependencies=[Depends(get_current_active_user)]
)
async def scrape_endpoint(
    start: int = Query(..., ge=1970, le=2023, description="Ano inicial (até 2023)"),
    end: int = Query(..., ge=1970, le=2023, description="Ano final (até 2023)"),
    tabs: List[str] = Query(
        ...,
        description="Caminho de abas e subabas, ex: ['Produção','Vinhos de mesa']"
    )
):
    """
    Retorna dados de scraping para as abas/subabas informadas,
    usando cache em SQLite para evitar buscas repetidas.
    """
    interval = (start, end)
    session = init_db("scraping.db")

    main_tab = tabs[0]
    subtab = "/".join(tabs[1:]) if len(tabs) > 1 else None

    cached = get_cached_years_db(session, main_tab, subtab)
    requested = set(range(start, end + 1))
    missing = sorted(requested - cached)

    if missing:
        for year in missing:
            df_new = await fetch_scrape((year, year), tabs)
            if not df_new.empty:
                save_scrape_results(session, main_tab, df_new, subtab=subtab)

    df_all = query_scrape_results(session, main_tab, interval, subtab=subtab)

    return {
        "count": len(df_all),
        "data": df_all.to_dict(orient="records")
    }
