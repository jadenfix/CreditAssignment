FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace/creditlab
COPY pyproject.toml README.md /workspace/creditlab/
COPY src /workspace/creditlab/src
COPY configs /workspace/creditlab/configs
COPY benchmarks /workspace/creditlab/benchmarks
COPY docs /workspace/creditlab/docs
COPY paper /workspace/creditlab/paper

RUN pip install --no-cache-dir .

ENTRYPOINT ["python", "-m", "creditlab.cli.main"]
