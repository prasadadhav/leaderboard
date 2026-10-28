FROM python:3.12-slim

WORKDIR /app

# Install torch CPU-only first (lighter than the GPU default, ~800 MB)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
# Remove torch from requirements so it isn't reinstalled (already installed above)
RUN grep -v "^torch" requirements.txt > requirements_notorch.txt \
    && pip install --no-cache-dir -r requirements_notorch.txt \
    && rm requirements_notorch.txt

COPY . .

EXPOSE 8765

# DB path overridden at runtime via LETZEVAL_DB_PATH env var
CMD ["python", "agent.py"]
