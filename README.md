# AI Gateway

Service Go en architecture hexagonale pour exposer une API IA unique avec fallback entre plusieurs modèles et providers.

## Démarrage

```bash
cp .env.example .env
# Renseigner au moins une clé: GEMINI_API_KEY, GROQ_API_KEY ou MISTRAL_API_KEY.
go run ./cmd/api
```

```bash
curl -s http://localhost:8080/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Explique le pattern repository en Go en 5 lignes.",
    "system": "Tu réponds en français.",
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

| Variable | Défaut | Description |
| --- | --- | --- |
| `GEMINI_API_KEY` | optionnel | Clé API Google Gemini. |
| `GROQ_API_KEY` | optionnel | Clé API Groq. |
| `MISTRAL_API_KEY` | optionnel | Clé API Mistral. |
| `SERVER_ADDR` | `:8080` | Adresse d'écoute HTTP. |
| `REQUEST_TIMEOUT_SECONDS` | `45` | Timeout par tentative modèle. |
| `GEMINI_API_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta` | Base URL Gemini. |
| `GROQ_API_BASE_URL` | `https://api.groq.com/openai/v1` | Base URL Groq compatible OpenAI. |
| `MISTRAL_API_BASE_URL` | `https://api.mistral.ai/v1` | Base URL Mistral compatible OpenAI. |
| `MODEL_FALLBACKS` | voir `.env.example` | Candidats testés dans l'ordre, au format `provider:model`. |

`GEMINI_MODEL_FALLBACKS` reste accepté comme compatibilité si `MODEL_FALLBACKS` est absent.

## Modèles JSON structurés validés

Tests courts effectués avec les clés locales :

| Provider | Modèles viables | Notes |
| --- | --- | --- |
| Gemini | `gemini-3.6-flash`, `gemini-3.5-flash`, `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`, `gemma-4-31b-it`, `gemma-4-26b-a4b-it` | JSON structuré via `responseMimeType` + `responseSchema`. `gemini-3.7-flash` a répondu `503` au test, donc pas mis dans le fallback par défaut. |
| Groq | `openai/gpt-oss-120b`, `openai/gpt-oss-20b` | JSON Schema strict validé via `response_format`. |
| Mistral | `mistral-small-latest`, `mistral-medium-latest`, `mistral-large-latest`, `ministral-8b-latest`, `ministral-3b-latest` | JSON Schema strict validé via `response_format`. Par défaut, on privilégie les modèles économiques/free-tier friendly. |

Priorité par défaut :

```text
gemini:gemini-3.6-flash
gemini:gemini-3.5-flash
gemini:gemma-4-31b-it
gemini:gemma-4-26b-a4b-it
gemini:gemini-3.5-flash-lite
gemini:gemini-3.1-flash-lite
groq:openai/gpt-oss-120b
groq:openai/gpt-oss-20b
mistral:mistral-small-latest
mistral:ministral-8b-latest
mistral:ministral-3b-latest
```

## Endpoints

### `GET /healthz`

Retourne l'état du service.

### `POST /v1/generate`

Requête :

```json
{
  "prompt": "Explique-moi...",
  "system": "Tu es un assistant...",
  "temperature": 0.7,
  "max_output_tokens": 1024,
  "response_schema": {
    "type": "object",
    "properties": {
      "answer": { "type": "string" }
    },
    "required": ["answer"],
    "additionalProperties": false
  }
}
```

Réponse :

```json
{
  "model": "gemini:gemini-3.6-flash",
  "text": "{\"answer\":\"...\"}",
  "fallback_used": false,
  "attempts": [
    {
      "model": "gemini:gemini-3.6-flash",
      "status": "success",
      "latency_ms": 1820
    }
  ]
}
```

## Architecture

- `cmd/api` : bootstrap HTTP.
- `internal/domain` : types métier, port `AIProvider`, erreurs.
- `internal/application` : orchestration et fallback.
- `internal/adapters/inbound/httpapi` : API REST.
- `internal/adapters/outbound/gemini` : adapter REST Gemini.
- `internal/adapters/outbound/openai` : adapter compatible OpenAI pour Groq et Mistral.
- `internal/adapters/outbound/router` : routage `provider:model`.
- `internal/config` : `.env` et configuration.
- `internal/platform/logger` : logger applicatif.

Le fallback est déclenché sur les erreurs récupérables : quotas `429`, timeouts, annulations, modèle indisponible et erreurs `5xx`. Les erreurs de payload, d'authentification et de sécurité ne déclenchent pas de fallback.
