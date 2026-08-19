# AI Gateway

[![CI](https://github.com/axelfrache/ai-gateway/actions/workflows/ci.yml/badge.svg)](https://github.com/axelfrache/ai-gateway/actions/workflows/ci.yml)
[![Go](https://img.shields.io/badge/Go-1.26-00ADD8?logo=go&logoColor=white)](https://go.dev/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![GHCR](https://img.shields.io/badge/GHCR-Published-24292F?logo=github&logoColor=white)](https://github.com/axelfrache/ai-gateway/pkgs/container/ai-gateway)
[![Gemini](https://img.shields.io/badge/Gemini-API-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)
[![Groq](https://img.shields.io/badge/Groq-API-F55036)](https://groq.com/)
[![Mistral](https://img.shields.io/badge/Mistral-API-FA520F)](https://mistral.ai/)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-API-111111)](https://openrouter.ai/)

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
| `internal/adapters/outbound/openai` | OpenAI-compatible adapter for Groq, Mistral, and OpenRouter |
| `internal/adapters/outbound/mcp` | MCP Streamable HTTP tool client |
| `internal/adapters/outbound/router` | `provider:model` routing |
| `internal/config` | Environment loading and configuration |

## Providers

| Provider | Validated structured JSON models |
|----------|----------------------------------|
| Gemini | `gemini-3.6-flash`, `gemini-3.5-flash`, `gemma-4-31b-it`, `gemma-4-26b-a4b-it`, `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite` |
| Groq | `openai/gpt-oss-120b`, `openai/gpt-oss-20b` |
| Mistral | `mistral-small-latest`, `mistral-medium-latest`, `mistral-large-latest`, `ministral-8b-latest`, `ministral-3b-latest` |
| OpenRouter | `openrouter/free` |

## Tool Calling

`/v1/chat/completions` accepts OpenAI-compatible `tools`, `tool_choice`, assistant `tool_calls`, and `tool` result messages. When a request includes tools and no explicit model is set, AI Gateway uses the dedicated tool-capable fallback chain:

```text
gemini:gemini-3.6-flash
gemini:gemini-3.5-flash
groq:openai/gpt-oss-120b
groq:openai/gpt-oss-20b
gemini:gemini-3.5-flash-lite
gemini:gemini-3.1-flash-lite
mistral:mistral-small-latest
mistral:ministral-8b-latest
mistral:ministral-3b-latest
openrouter:openrouter/free
openrouter:z-ai/glm-5.2:free
openrouter:google/gemma-4-31b-it:free
openrouter:google/gemma-4-26b-a4b-it:free
openrouter:openai/gpt-oss-20b:free
```

When `MCP_SERVERS` is configured, AI Gateway lists tools from each MCP server, injects them into chat completion requests, executes selected tool calls, appends tool results, and calls the model again until a final assistant response is produced.

MCP tools are exposed as `server__tool_name`. For example, a Kubernetes MCP server named `kube` with a `get-pods` tool becomes `kube__get_pods`.

```env
MCP_SERVERS=kube=http://kube-mcp.ai.svc.cluster.local/mcp
MCP_BEARER_TOKENS=kube=optional-token
MCP_ALLOWED_TOOLS=kube__get*,kube__describe*
MCP_DENIED_TOOLS=kube__get_secret*
MCP_MAX_TOOL_ROUNDS=4
MCP_TOOL_TIMEOUT_SECONDS=20
```

AI Gateway currently supports MCP Streamable HTTP JSON-RPC endpoints.

## Getting Started

### Prerequisites

- Go 1.26
- Docker and Docker Compose
- At least one provider key: `GEMINI_API_KEY`, `GROQ_API_KEY`, `MISTRAL_API_KEY`, or `OPENROUTER_API_KEY`
- At least one gateway key in `GATEWAY_API_KEYS`

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
- Models API: http://localhost:8080/v1/models
- OpenAI-compatible chat API: http://localhost:8080/v1/chat/completions
- Ollama-compatible generate API: http://localhost:8080/api/generate
- Ollama-compatible models API: http://localhost:8080/api/tags

To stop:

```bash
docker compose down
```

## API

```bash
curl -s http://localhost:8080/v1/generate \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer change-me' \
  -d '{
    "prompt": "Explain the repository pattern in Go in 5 lines.",
    "system": "Answer in English.",
    "model": "gemini:gemini-3.6-flash",
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

Use `model` to force a single model, or `models` to provide a request-specific fallback chain:

```json
{
  "models": [
    "gemini:gemini-3.6-flash",
    "groq:openai/gpt-oss-20b"
  ]
}
```

List configured models:

```bash
curl -s http://localhost:8080/v1/models \
  -H 'Authorization: Bearer change-me'
```

Probe model availability:

```bash
curl -s http://localhost:8080/v1/models/check \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer change-me' \
  -d '{"models":["gemini:gemini-3.6-flash"]}'
```

## OpenAI-Compatible API

Use `/v1/chat/completions` with OpenAI-compatible clients and WebUIs:

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer change-me' \
  -d '{
    "model": "gemini:gemini-3.6-flash",
    "messages": [
      { "role": "system", "content": "Answer in English." },
      { "role": "user", "content": "Explain dependency injection in Go in 3 lines." }
    ],
    "temperature": 0.3,
    "max_tokens": 256
  }'
```

For Open WebUI, configure an OpenAI-compatible connection:

```env
OPENAI_API_BASE_URL=http://ai-gateway.ai-gateway.svc.cluster.local:8080/v1
OPENAI_API_KEY=change-me
```

`stream: true` is accepted and returned as a single server-sent event chunk.

## Ollama-Compatible API

AI Gateway also exposes a small Ollama-compatible surface for apps that expect Ollama-style endpoints.

List configured models without probing providers:

```bash
curl -s http://localhost:8080/api/tags \
  -H 'Authorization: Bearer change-me'
```

Generate a non-streaming response:

```bash
curl -s http://localhost:8080/api/generate \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer change-me' \
  -d '{
    "model": "gemini:gemini-3.6-flash",
    "prompt": "Return a JSON object with a short greeting.",
    "stream": false,
    "format": "json",
    "options": {
      "temperature": 0.2,
      "num_predict": 128
    }
  }'
```

`format` accepts `"json"` or a JSON Schema object. `stream: true` is rejected because AI Gateway currently returns one completed response per request.

## Configuration

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Google Gemini API key |
| `GROQ_API_KEY` | Groq API key |
| `MISTRAL_API_KEY` | Mistral API key |
| `OPENROUTER_API_KEY` | OpenRouter API key |
| `SERVER_ADDR` | HTTP listen address |
| `REQUEST_TIMEOUT_SECONDS` | Timeout per model attempt |
| `GATEWAY_API_KEYS` | Comma-separated API keys allowed to call protected endpoints |
| `MODEL_FALLBACKS` | Ordered fallback chain using `provider:model` |
| `TOOL_MODEL_FALLBACKS` | Ordered fallback chain for chat completion requests that include tools |
| `MCP_SERVERS` | Comma-separated MCP servers using `name=url`; empty disables gateway-side MCP tools |
| `MCP_BEARER_TOKENS` | Optional comma-separated MCP bearer tokens using `name=token` |
| `MCP_ALLOWED_TOOLS` | Optional allow list for exposed MCP tools; supports exact names, prefix `*`, and suffix `*` |
| `MCP_DENIED_TOOLS` | Optional deny list applied before the allow list |
| `MCP_PROTOCOL_VERSION` | MCP protocol version sent during initialization |
| `MCP_MAX_TOOL_ROUNDS` | Maximum assistant-tool loops per chat request |
| `MCP_TOOL_TIMEOUT_SECONDS` | Timeout per MCP HTTP request |

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
