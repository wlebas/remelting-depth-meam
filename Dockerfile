FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir numpy scipy pandas numba openpyxl

COPY *.py ./
COPY *.csv ./

ENTRYPOINT ["python"]
CMD ["dimensionless_sweep_pure.py"]