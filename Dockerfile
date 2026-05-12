FROM python:3.13-slim-bullseye

ARG GIT_SHA=unknown
ENV WYOMING_MS_TTS_GIT_SHA=$GIT_SHA

# Install the Python package
COPY . /app
WORKDIR /app
RUN pip install --no-cache-dir .

EXPOSE 10200

ENTRYPOINT [ "python", "-m", "wyoming_microsoft_tts"]