# AI Gateway

[![CI](https://github.com/axelfrache/ai-gateway/actions/workflows/ci.yml/badge.svg)](https://github.com/axelfrache/ai-gateway/actions/workflows/ci.yml)
[![Go](https://img.shields.io/badge/Go-1.26-00ADD8?logo=go&logoColor=white)](https://go.dev/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![GHCR](https://img.shields.io/badge/GHCR-Published-24292F?logo=github&logoColor=white)](https://github.com/axelfrache/ai-gateway/pkgs/container/ai-gateway)
[![Gemini](https://img.shields.io/badge/Gemini-API-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)
[![Groq](https://img.shields.io/badge/Groq-API-F55036)](https://groq.com/)
[![Mistral](https://img.shields.io/badge/Mistral-API-FA520F)](https://mistral.ai/)

## Description

AI Gateway is a Go service that exposes a single AI API with fallback across multiple providers and models.

It is designed for structured JSON responses, free-tier-friendly routing, and a clean hexagonal architecture.

## Architecture

| Layer | Role |
|-------|------|
| `cmd/api` | HTTP bootstrap |
| `internal/domain` | Domain types, provider port, errors |
| `internal/application` | Generation orchestration and fallback |
| `internal/adapters/inbound/httpapi` | REST API |
| `internal/adapters/outbound/gemini` | Gemini adapter |
| `internal/adapters/outbound/openai` | OpenAI-compatible adapter for Groq and Mistral |
| `internal/adapters/outbound/router` | `provider:model` routing |
| `internal/config` | Environment loading and configuration |

## Providers

| Provider | Validated structured JSON models |
|----------|----------------------------------|
| Gemini | `gemini-3.6-flash`, `gemini-3.5-flash`, `gemma-4-31b-it`, `gemma-4-26b-a4b-it`, `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite` |
| Groq | `openai/gpt-oss-120b`, `openai/gpt-oss-20b` |
| Mistral | `mistral-small-latest`, `mistral-medium-latest`, `mistral-large-latest`, `ministral-8b-latest`, `ministral-3b-latest` |

## Getting Started

### Prerequisites

- Go 1.26
- Docker and Docker Compose
- At least one provider key: `GEMINI_API_KEY`, `GROQ_API_KEY`, or `MISTRAL_API_KEY`

## Running

### Local

```bash
cp .env.example .env
go run ./cmd/api
```

### Docker Compose

```bash
docker compose up -d --build
```

Then go to:

- Health check: http://localhost:8080/healthz
- Generate API: http://localhost:8080/v1/generate

To stop:

```bash
docker compose down
```

## API

```bash
curl -s http://localhost:8080/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Explain the repository pattern in Go in 5 lines.",
    "system": "Answer in English.",
    "temperature": 0.4,
    "max_output_tokens": 512,
    "response_schema": {
      "type": "object",
      "properties": {
        "summary": { "type": "string" }
      },
      "required": ["summary"],
      "additionalProperties": false
    }
  }'
```

## Configuration

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Google Gemini API key |
| `GROQ_API_KEY` | Groq API key |
| `MISTRAL_API_KEY` | Mistral API key |
| `SERVER_ADDR` | HTTP listen address |
| `REQUEST_TIMEOUT_SECONDS` | Timeout per model attempt |
| `MODEL_FALLBACKS` | Ordered fallback chain using `provider:model` |

## Code Quality

Formatting, vetting, tests, binary build, Docker build, and GHCR publishing are handled by CI.

### Commands

```bash
test -z "$(gofmt -l .)"
go vet ./...
go test ./...
go build -v -o /tmp/ai-gateway ./cmd/api
docker build -t ai-gateway:local .
```
