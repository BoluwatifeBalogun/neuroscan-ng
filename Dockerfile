FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt tensorflow-cpu gunicorn
COPY . .
RUN mkdir -p uploads model && chmod -R 777 uploads model .
EXPOSE 7860
CMD ["gunicorn", "-b", "0.0.0.0:7860", "-w", "2", "--timeout", "120", "app:app"]
