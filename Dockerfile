FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY . .

RUN cd csrc && python setup.py build_ext --inplace 2>/dev/null; \
    echo "C++ extension: $([ -f logit_processors*.so ] && echo built || echo skipped)"

RUN python -m compileall . -q 2>/dev/null; echo "Bytecode compiled"

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "start.py"]
CMD ["--help"]
