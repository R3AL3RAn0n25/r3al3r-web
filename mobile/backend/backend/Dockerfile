FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["waitress-serve", "--host=0.0.0.0", "--port=5000", "--threads=4", "--call", "r3al3r_ai_framework:app"]
