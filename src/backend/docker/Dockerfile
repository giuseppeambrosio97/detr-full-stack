# requirements
FROM python:3.11-slim as requirements-stage
WORKDIR /tmp

ENV PIP_ROOT_USER_ACTION=ignore

RUN pip install poetry==1.7.1
COPY ./pyproject.toml ./poetry.lock /tmp/
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes --without dev,test --without-urls

# build
FROM python:3.11 as build-stage
WORKDIR /app

ENV PIP_ROOT_USER_ACTION=ignore

COPY --from=requirements-stage /tmp/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
COPY ./backend /app/backend

WORKDIR /app
# CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8080"]
CMD ["python", "-m", "backend"]