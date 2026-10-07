# ─────────────────────────────────────────────────────────────────────────────
# Argus - one command to start, one to stop.
#
#   make          show this help
#   make start    set up and run everything
#   make stop     stop everything
#
# These are thin wrappers over `python argus.py`, which is the real launcher and
# works identically on Windows without make. The Docker targets exist because
# `docker run` otherwise requires the user to invent a JWT secret by hand.
# ─────────────────────────────────────────────────────────────────────────────

PY      ?= python3
COMPOSE ?= docker compose
PROD    := docker-compose.prod.yml

.DEFAULT_GOAL := help
.PHONY: help start stop restart status doctor logs reset test \
        docker-start docker-stop docker-logs build

help:
	@echo ""
	@echo "  Argus"
	@echo ""
	@echo "  make start          set up and run everything (venv, deps, DB, UI, server)"
	@echo "  make stop           stop everything"
	@echo "  make restart        stop then start"
	@echo "  make status         show what is running"
	@echo "  make logs           follow the server log"
	@echo ""
	@echo "  make docker-start   same, but in a container (needs Docker)"
	@echo "  make docker-stop    stop the container"
	@echo ""
	@echo "  make doctor         check this machine can run Argus"
	@echo "  make test           run the test suite"
	@echo "  make reset          stop and delete generated files"
	@echo ""
	@echo "  Without make (any OS):  $(PY) argus.py start  |  $(PY) argus.py stop"
	@echo ""

start:
	@$(PY) argus.py start

stop:
	@$(PY) argus.py stop

restart:
	@$(PY) argus.py stop
	@$(PY) argus.py start

status:
	@$(PY) argus.py status

doctor:
	@$(PY) argus.py doctor

logs:
	@tail -f .argus/backend.log

reset:
	@$(PY) argus.py reset

test:
	@$(PY) -m pytest -q

# ── Docker ───────────────────────────────────────────────────────────────────
# Generates .env with a real signing key on first run. Without it the compose
# file's `:?` guard aborts with a message about a variable the user was never
# told to set - technically correct, unhelpful as a first experience.
.env:
	@cp .env.example .env
	@$(PY) -c "import secrets,pathlib; p=pathlib.Path('.env'); \
	  t=p.read_text().replace('ARGUS_JWT_SECRET=', 'ARGUS_JWT_SECRET='+secrets.token_urlsafe(48), 1); \
	  p.write_text(t)"
	@echo "  Created .env with a generated ARGUS_JWT_SECRET"

docker-start: .env
	@$(COMPOSE) -f $(PROD) up -d --build
	@echo ""
	@echo "  Argus is starting - first boot loads the detector, give it ~60s."
	@echo "  Dashboard   http://localhost:8000"
	@echo "  Login       admin (set via ARGUS_ADMIN_PASSWORD in .env or python argus.py create-admin)"
	@echo "  Logs        make docker-logs"
	@echo "  Stop        make docker-stop"
	@echo ""

docker-stop:
	@$(COMPOSE) -f $(PROD) down
	@echo "  Stopped."

docker-logs:
	@$(COMPOSE) -f $(PROD) logs -f

build:
	@docker build -t argus .
