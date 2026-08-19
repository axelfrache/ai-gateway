package config

import (
	"errors"
	"os"
	"strconv"
	"strings"
	"time"
)

const defaultGeminiAPIBaseURL = "https://generativelanguage.googleapis.com/v1beta"
const defaultGroqAPIBaseURL = "https://api.groq.com/openai/v1"
const defaultMistralAPIBaseURL = "https://api.mistral.ai/v1"
const defaultOpenRouterAPIBaseURL = "https://openrouter.ai/api/v1"

var defaultModelFallbacks = []string{
	"gemini:gemini-3.6-flash",
	"gemini:gemini-3.5-flash",
	"gemini:gemma-4-31b-it",
	"gemini:gemma-4-26b-a4b-it",
	"gemini:gemini-3.5-flash-lite",
	"gemini:gemini-3.1-flash-lite",
	"groq:openai/gpt-oss-120b",
	"groq:openai/gpt-oss-20b",
	"mistral:mistral-small-latest",
	"mistral:ministral-8b-latest",
	"mistral:ministral-3b-latest",
	"openrouter:openrouter/free",
}

var defaultToolModelFallbacks = []string{
	"gemini:gemini-3.6-flash",
	"gemini:gemini-3.5-flash",
	"groq:openai/gpt-oss-120b",
	"groq:openai/gpt-oss-20b",
	"gemini:gemini-3.5-flash-lite",
	"gemini:gemini-3.1-flash-lite",
	"mistral:mistral-small-latest",
	"mistral:ministral-8b-latest",
	"mistral:ministral-3b-latest",
	"openrouter:openrouter/free",
	"openrouter:z-ai/glm-5.2:free",
	"openrouter:google/gemma-4-31b-it:free",
	"openrouter:google/gemma-4-26b-a4b-it:free",
	"openrouter:openai/gpt-oss-20b:free",
}

type Config struct {
	ServerAddr         string
	GatewayAPIKeys     []string
	GeminiAPIKey       string
	GeminiBaseURL      string
	GroqAPIKey         string
	GroqBaseURL        string
	MistralAPIKey      string
	MistralBaseURL     string
	OpenRouterAPIKey   string
	OpenRouterBaseURL  string
	ModelFallbacks     []string
	ToolModelFallbacks []string
	MCPServers         []MCPServer
	MCPAllowedTools    []string
	MCPDeniedTools     []string
	MCPProtocolVersion string
	MCPMaxToolRounds   int
	MCPToolTimeout     time.Duration
	RequestTimeout     time.Duration
}

type MCPServer struct {
	Name        string
	URL         string
	BearerToken string
}

func FromEnv() (Config, error) {
	timeoutSeconds, err := intFromEnv("REQUEST_TIMEOUT_SECONDS", 45)
	if err != nil {
		return Config{}, err
	}
	mcpMaxToolRounds, err := intFromEnv("MCP_MAX_TOOL_ROUNDS", 4)
	if err != nil {
		return Config{}, err
	}
	mcpToolTimeoutSeconds, err := intFromEnv("MCP_TOOL_TIMEOUT_SECONDS", 20)
	if err != nil {
		return Config{}, err
	}

	mcpBearerTokens, err := pairsFromEnv("MCP_BEARER_TOKENS")
	if err != nil {
		return Config{}, err
	}
	mcpServers, err := mcpServersFromEnv("MCP_SERVERS", mcpBearerTokens)
	if err != nil {
		return Config{}, err
	}

	cfg := Config{
		ServerAddr:       stringFromEnv("SERVER_ADDR", ":8080"),
		GatewayAPIKeys:   csvFromEnv("GATEWAY_API_KEYS", csvFromEnv("GATEWAY_API_KEY", nil)),
		GeminiAPIKey:     os.Getenv("GEMINI_API_KEY"),
		GeminiBaseURL:    strings.TrimRight(stringFromEnv("GEMINI_API_BASE_URL", defaultGeminiAPIBaseURL), "/"),
		GroqAPIKey:       os.Getenv("GROQ_API_KEY"),
		GroqBaseURL:      strings.TrimRight(stringFromEnv("GROQ_API_BASE_URL", defaultGroqAPIBaseURL), "/"),
		MistralAPIKey:    os.Getenv("MISTRAL_API_KEY"),
		MistralBaseURL:   strings.TrimRight(stringFromEnv("MISTRAL_API_BASE_URL", defaultMistralAPIBaseURL), "/"),
		OpenRouterAPIKey: os.Getenv("OPENROUTER_API_KEY"),
		OpenRouterBaseURL: strings.TrimRight(
			stringFromEnv("OPENROUTER_API_BASE_URL", defaultOpenRouterAPIBaseURL),
			"/",
		),
		ModelFallbacks:     csvFromEnv("MODEL_FALLBACKS", csvFromEnv("GEMINI_MODEL_FALLBACKS", defaultModelFallbacks)),
		ToolModelFallbacks: csvFromEnv("TOOL_MODEL_FALLBACKS", defaultToolModelFallbacks),
		MCPServers:         mcpServers,
		MCPAllowedTools:    csvFromEnv("MCP_ALLOWED_TOOLS", nil),
		MCPDeniedTools:     csvFromEnv("MCP_DENIED_TOOLS", nil),
		MCPProtocolVersion: stringFromEnv("MCP_PROTOCOL_VERSION", "2025-11-25"),
		MCPMaxToolRounds:   mcpMaxToolRounds,
		MCPToolTimeout:     time.Duration(mcpToolTimeoutSeconds) * time.Second,
		RequestTimeout:     time.Duration(timeoutSeconds) * time.Second,
	}

	if cfg.GeminiAPIKey == "" && cfg.GroqAPIKey == "" && cfg.MistralAPIKey == "" && cfg.OpenRouterAPIKey == "" {
		return Config{}, errors.New("at least one provider API key is required")
	}
	if len(cfg.GatewayAPIKeys) == 0 {
		return Config{}, errors.New("GATEWAY_API_KEYS is required")
	}

	return cfg, nil
}

func stringFromEnv(key, fallback string) string {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return fallback
	}
	return value
}

func intFromEnv(key string, fallback int) (int, error) {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return fallback, nil
	}
	parsed, err := strconv.Atoi(value)
	if err != nil {
		return 0, err
	}
	return parsed, nil
}

func csvFromEnv(key string, fallback []string) []string {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return append([]string(nil), fallback...)
	}

	parts := strings.Split(value, ",")
	result := make([]string, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part != "" {
			result = append(result, part)
		}
	}
	if len(result) == 0 {
		return append([]string(nil), fallback...)
	}
	return result
}

func pairsFromEnv(key string) (map[string]string, error) {
	pairs := map[string]string{}
	for _, entry := range csvFromEnv(key, nil) {
		name, value, ok := strings.Cut(entry, "=")
		if !ok {
			return nil, errors.New(key + " entries must use name=value")
		}
		name = strings.TrimSpace(name)
		value = strings.TrimSpace(value)
		if name == "" || value == "" {
			return nil, errors.New(key + " entries must use non-empty name=value")
		}
		pairs[name] = value
	}
	return pairs, nil
}

func mcpServersFromEnv(key string, bearerTokens map[string]string) ([]MCPServer, error) {
	entries := csvFromEnv(key, nil)
	servers := make([]MCPServer, 0, len(entries))
	for _, entry := range entries {
		name, url, ok := strings.Cut(entry, "=")
		if !ok {
			return nil, errors.New(key + " entries must use name=url")
		}
		name = strings.TrimSpace(name)
		url = strings.TrimSpace(url)
		if name == "" || url == "" {
			return nil, errors.New(key + " entries must use non-empty name=url")
		}
		servers = append(servers, MCPServer{
			Name:        name,
			URL:         strings.TrimRight(url, "/"),
			BearerToken: bearerTokens[name],
		})
	}
	return servers, nil
}
