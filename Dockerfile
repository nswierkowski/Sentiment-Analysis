FROM python:3.12

WORKDIR /app

COPY requirements.txt .
RUN pip --no-cache-dir install -r requirements.txt

# Create a non-root user matching the host user
ARG USER_ID=1000
ARG GROUP_ID=1000

RUN groupadd -g ${GROUP_ID} appgroup && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} appuser

# Ensure proper ownership
RUN chown -R appuser:appgroup /app

# Switch to this user
USER appuser

ENV PYTHONPATH=/app

EXPOSE 8888
