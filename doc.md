# Projeto: **projeto3** — Scraper + API + ML (Logística) + Stack Docker

Documentação curta (e direta) para subir, usar e testar o projeto.

---

## Visão geral

- **Scraper** do site Embrapa (Vitibrasil) via `requests + BeautifulSoup`.
- **API FastAPI** com **JWT** (criar usuário, obter token, liberar endpoints).
- **Persistência** em **PostgreSQL** (scrapes → `scrape_records`, dataset tratado → `ml_dataset`, métricas → `ml_metrics`).
- **Job de ML** (Regressão Logística) treinado via endpoint; rótulo multiclasse baseado em **média móvel (3 anos)** e **índice composto + quantis** (evita classe única).
- **Stack Docker** com `postgres`, `pgAdmin`, `grafana`, `projeto3`.

---

## Estrutura de pastas

```
projeto3/
├─ docker-compose.yml
├─ requirements.txt
├─ init/
│  ├─ 00_enable_dblink.sql        # opcional (se optar por dblink)
│  └─ 01_grafana.sql / 01_grafana.sh  # cria DB/usuário para Grafana (init do Postgres)
└─ src/
   ├─ api/
   │  └─ app.py                   # FastAPI (endpoints /scrape, /ml/run, /users, /token)
   ├─ db_querys/
   │  ├─ db_utils.py              # ScrapeRecord + helpers (Postgres)
   │  └─ db_users.py              # User model + CRUD (Postgres)
   ├─ job/
   │  └─ ml_job.py                # Pipeline ML (run_and_save_ml)
   └─ utils/
      └─ auth.py                  # JWT + rotas de auth (sempre Postgres)
```

---

## Variáveis de ambiente importantes

- `DATABASE_URL` (usada pela API e pelo job):

  - `postgresql://user:password@db:5432/projeto3`

- `SECRET_KEY` (JWT) — **defina em produção**.
- `JWT_EXPIRE_MINUTES` (opcional, padrão `60`).

---

## Subir com Docker

1. **Pré-requisitos**

   - Docker & Docker Compose
   - `requirements.txt` inclui `psycopg2-binary` (driver Postgres)

2. **Subir**

```bash
docker compose up -d --build
```

3. **Acessos**

- API: [http://localhost:8000/docs](http://localhost:8000/docs)
- pgAdmin: [http://localhost:5050](http://localhost:5050)

  - Email: `admin@projeto3.local`
  - Senha: `admin`
  - Adicione um servidor apontando para **Host**: `db`, **Port**: `5432`, **User**: `user`, **Pass**: `password`, **DB**: `projeto3`

- Grafana: [http://localhost:3050](http://localhost:3050) (se configurado no compose)

  - Admin: `admin` / `admin` (ou conforme envs)

> Se estiver usando Postgres como backend do Grafana, use os scripts de init para criar `grafana`/`grafana`.

---

## Autenticação (JWT)

Rotas protegidas exigem **Bearer token**:

- `POST /users` — cria usuário
- `POST /token` — retorna `{access_token, token_type}`

Exemplo cURL:

```bash
# criar usuário (uma vez)
curl -X POST http://localhost:8000/users \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin","full_name":"Admin"}'

# obter token
TOKEN=$(curl -s -X POST http://localhost:8000/token \
  -d "username=admin&password=admin" | python -c "import sys,json;print(json.load(sys.stdin)['access_token'])")
echo $TOKEN
```

---

## Endpoints principais

### `GET /scrape` (protegido)

Faz scraping (com cache no Postgres por ano/tab) e retorna JSON.

**Query params:**

- `start` (int), `end` (int)
- `tabs` (repetir para cada nível), ex.: `tabs=Produção&tabs=Vinhos de mesa`

**Exemplo:**

```bash
curl -G "http://localhost:8000/scrape" \
  -H "Authorization: Bearer $TOKEN" \
  --data-urlencode "start=2018" \
  --data-urlencode "end=2019" \
  --data-urlencode "tabs=Produção" \
  --data-urlencode "tabs=Vinhos de mesa"
```

### `POST /ml/run` (protegido)

Executa o pipeline de ML e **salva**:

- `ml_dataset` (dataset tratado)
- `ml_metrics` (métricas da última execução)

Retorna **204 No Content**.

```bash
curl -X POST "http://localhost:8000/ml/run" \
  -H "Authorization: Bearer $TOKEN" -i
```

---

## Pipeline de ML (resumo)

Arquivo: `src/job/ml_job.py`

- Leitura: `scrape_records`
- Agregação anual:

  - **Produção**: **apenas** “Vinho de Mesa” (não o total)
  - **Exportação**: total em **US$**

- Rótulo `desempenho` (multiclasse) — **opção recomendada**:

  - Média móvel dos **3 anos anteriores**
  - Desvios relativos (produção/exportação) vs médias
  - Índice composto (`score = 0.5*prod_dev + 0.5*exp_dev`)
  - Discretização por **quantis** (tercis → classes `0,1,2`)

- Split (50/50, com fallback sem `stratify` se necessário)
- Treino: `LogisticRegression(multi_class="multinomial", solver="lbfgs")`
- **Guards**: se houver só 1 classe, aborta o treino sem erro e mantém o dataset salvo.

Consultar rápido via psql:

```bash
docker compose exec -T db psql -U user -d projeto3 -c "SELECT * FROM ml_dataset ORDER BY year DESC LIMIT 10;"
docker compose exec -T db psql -U user -d projeto3 -c "SELECT * FROM ml_metrics ORDER BY id DESC LIMIT 5;"
```

---

## Testando no Postman

1. Crie um **Environment** `projeto3` com:

   - `baseUrl = http://localhost:8000`
   - `username` / `password` (ex.: `admin` / `admin`)
   - `token` (vazio inicialmente)

2. **POST `{{baseUrl}}/users`** (Body JSON) → 201/400
3. **POST `{{baseUrl}}/token`** (Body `x-www-form-urlencoded`)
   Em **Tests**:

   ```js
   const data = pm.response.json();
   pm.environment.set('token', data.access_token);
   ```

4. Configure Authorization (Bearer) da **Collection** com `{{token}}`.
5. **GET `{{baseUrl}}/scrape`** com params:

   - `start=2018`, `end=2019`, `tabs=Produção`, `tabs=Vinhos de mesa`

6. **POST `{{baseUrl}}/ml/run`** → 204

---

## Desenvolvimento/Execução local (sem Docker)

```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
export DATABASE_URL="postgresql://user:password@localhost:5432/projeto3"
export SECRET_KEY="dev-secret"
uvicorn api.app:app --reload
```

---

## Dicas & Solução de problemas

- **Grafana não abre em 3050** → confira o mapeamento `3050:3000` no compose.
- **Grafana com “database is locked”** → limpe volume `grafana_data` **ou** use Postgres como backend do Grafana (scripts em `init/`).
- **pgAdmin e e-mail inválido** → use `admin@projeto3.com` **ou** habilite domínios especiais no compose.
- **Erro no ML (“apenas uma classe”)** → já há guard no pipeline. Garanta dados suficientes (rodar `/scrape` para mais anos) e use a **regra por quantis**.
- **Drivers Postgres** → `psycopg2-binary` deve estar no `requirements.txt`.

---

## Licença

Use à vontade para fins de estudo e extensão interna. Ajuste para seu contexto de produção (chaves, segurança, observabilidade, CI/CD).
