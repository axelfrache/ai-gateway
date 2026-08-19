package httpapi

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/frachea/ai-gateway/internal/application"
	"github.com/frachea/ai-gateway/internal/domain"
)

func TestGenerateEndpoint(t *testing.T) {
	service, err := application.NewAIService(fakeProvider{}, []string{"gemini-3.7-flash"})
	if err != nil {
		t.Fatal(err)
	}
	server := NewServer(service, slog.New(slog.DiscardHandler), []string{"test-key"}).Routes()

	body, err := json.Marshal(map[string]any{
		"prompt": "hello",
	})
	if err != nil {
		t.Fatal(err)
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/generate", bytes.NewReader(body))
	req.Header.Set("Authorization", "Bearer test-key")
	rec := httptest.NewRecorder()

	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", rec.Code, rec.Body.String())
	}

	var result domain.GenerateResult
	if err := json.Unmarshal(rec.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result.Model != "gemini-3.7-flash" {
		t.Fatalf("expected model gemini-3.7-flash, got %q", result.Model)
	}
	if result.Text != "ok" {
		t.Fatalf("expected text ok, got %q", result.Text)
	}
}

func TestGenerateEndpointRequiresAPIKey(t *testing.T) {
	service, err := application.NewAIService(fakeProvider{}, []string{"gemini-3.7-flash"})
	if err != nil {
		t.Fatal(err)
	}
	server := NewServer(service, slog.New(slog.DiscardHandler), []string{"test-key"}).Routes()

	body, err := json.Marshal(map[string]any{
		"prompt": "hello",
	})
	if err != nil {
		t.Fatal(err)
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/generate", bytes.NewReader(body))
	rec := httptest.NewRecorder()

	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("expected status 401, got %d: %s", rec.Code, rec.Body.String())
	}
}

func TestGenerateEndpointAcceptsXAPIKey(t *testing.T) {
	service, err := application.NewAIService(fakeProvider{}, []string{"gemini-3.7-flash"})
	if err != nil {
		t.Fatal(err)
	}
	server := NewServer(service, slog.New(slog.DiscardHandler), []string{"test-key"}).Routes()

	body, err := json.Marshal(map[string]any{
		"prompt": "hello",
	})
	if err != nil {
		t.Fatal(err)
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/generate", bytes.NewReader(body))
	req.Header.Set("X-API-Key", "test-key")
	rec := httptest.NewRecorder()

	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", rec.Code, rec.Body.String())
	}
}

func TestGenerateEndpointAcceptsModelOverride(t *testing.T) {
	service, err := application.NewAIService(fakeProvider{}, []string{"gemini-3.7-flash"})
	if err != nil {
		t.Fatal(err)
	}
	server := NewServer(service, slog.New(slog.DiscardHandler), []string{"test-key"}).Routes()

	body, err := json.Marshal(map[string]any{
		"prompt": "hello",
		"model":  "groq:openai/gpt-oss-20b",
	})
	if err != nil {
		t.Fatal(err)
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/generate", bytes.NewReader(body))
	req.Header.Set("Authorization", "Bearer test-key")
	rec := httptest.NewRecorder()

	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", rec.Code, rec.Body.String())
	}

	var result domain.GenerateResult
	if err := json.Unmarshal(rec.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result.Model != "groq:openai/gpt-oss-20b" {
		t.Fatalf("expected override model, got %q", result.Model)
	}
}

func TestModelsEndpoint(t *testing.T) {
	service, err := application.NewAIService(fakeProvider{}, []string{"gemini:gemini-3.6-flash", "groq:openai/gpt-oss-20b"})
	if err != nil {
		t.Fatal(err)
	}
	server := NewServer(service, slog.New(slog.DiscardHandler), []string{"test-key"}).Routes()

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	req.Header.Set("Authorization", "Bearer test-key")
	rec := httptest.NewRecorder()

	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", rec.Code, rec.Body.String())
	}

	var result struct {
		Models []domain.ModelInfo `json:"models"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if len(result.Models) != 2 {
		t.Fatalf("expected 2 models, got %d", len(result.Models))
	}
	if result.Models[0].Provider != "gemini" || result.Models[0].Model != "gemini-3.6-flash" {
		t.Fatalf("unexpected first model: %#v", result.Models[0])
	}
}

func TestModelsEndpointIncludesOpenAIShape(t *testing.T) {
	service, err := application.NewAIService(fakeProvider{}, []string{"gemini:gemini-3.6-flash"})
	if err != nil {
		t.Fatal(err)
	}
	server := NewServer(service, slog.New(slog.DiscardHandler), []string{"test-key"}).Routes()

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	req.Header.Set("Authorization", "Bearer test-key")
	rec := httptest.NewRecorder()

	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", rec.Code, rec.Body.String())
	}

	var result struct {
		Object string `json:"object"`
		Data   []struct {
			ID      string `json:"id"`
			Object  string `json:"object"`
			OwnedBy string `json:"owned_by"`
		} `json:"data"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result.Object != "list" {
		t.Fatalf("expected object list, got %q", result.Object)
	}
	if len(result.Data) != 1 || result.Data[0].ID != "gemini:gemini-3.6-flash" || result.Data[0].Object != "model" {
		t.Fatalf("unexpected OpenAI model data: %#v", result.Data)
	}
}

func TestCheckModelsEndpoint(t *testing.T) {
	service, err := application.NewAIService(fakeProvider{}, []string{"gemini:gemini-3.6-flash"})
	if err != nil {
		t.Fatal(err)
	}
	server := NewServer(service, slog.New(slog.DiscardHandler), []string{"test-key"}).Routes()

	body, err := json.Marshal(map[string]any{
		"models": []string{"groq:openai/gpt-oss-20b"},
	})
	if err != nil {
		t.Fatal(err)
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/models/check", bytes.NewReader(body))
	req.Header.Set("Authorization", "Bearer test-key")
	rec := httptest.NewRecorder()

	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", rec.Code, rec.Body.String())
	}

	var result struct {
		Models []domain.ModelCheck `json:"models"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if len(result.Models) != 1 {
		t.Fatalf("expected 1 model check, got %d", len(result.Models))
	}
	if result.Models[0].Status != "available" {
		t.Fatalf("expected available model, got %#v", result.Models[0])
	}
}

func TestOllamaTagsEndpoint(t *testing.T) {
	service, err := application.NewAIService(fakeProvider{}, []string{"gemini:gemini-3.6-flash", "groq:openai/gpt-oss-20b"})
	if err != nil {
		t.Fatal(err)
	}
	server := NewServer(service, slog.New(slog.DiscardHandler), []string{"test-key"}).Routes()

	req := httptest.NewRequest(http.MethodGet, "/api/tags", nil)
	req.Header.Set("Authorization", "Bearer test-key")
	rec := httptest.NewRecorder()

	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", rec.Code, rec.Body.String())
	}

	var result struct {
		Models []struct {
			Name    string `json:"name"`
			Model   string `json:"model"`
			Details struct {
				Format   string   `json:"format"`
				Family   string   `json:"family"`
				Families []string `json:"families"`
			} `json:"details"`
		} `json:"models"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if len(result.Models) != 2 {
		t.Fatalf("expected 2 models, got %d", len(result.Models))
	}
	if result.Models[0].Name != "gemini:gemini-3.6-flash" || result.Models[0].Model != "gemini:gemini-3.6-flash" {
		t.Fatalf("unexpected first model: %#v", result.Models[0])
	}
	if result.Models[0].Details.Format != "remote" || result.Models[0].Details.Family != "gemini" {
		t.Fatalf("unexpected model details: %#v", result.Models[0].Details)
	}
}

func TestOllamaGenerateEndpoint(t *testing.T) {
	provider := &capturingProvider{}
	service, err := application.NewAIService(provider, []string{"gemini:gemini-3.6-flash"})
	if err != nil {
		t.Fatal(err)
	}
	server := NewServer(service, slog.New(slog.DiscardHandler), []string{"test-key"}).Routes()

	stream := false
	body, err := json.Marshal(map[string]any{
		"model":  "groq:openai/gpt-oss-20b",
		"prompt": "hello",
		"system": "answer briefly",
		"stream": stream,
		"format": map[string]any{
			"type": "object",
			"properties": map[string]any{
				"answer": map[string]any{"type": "string"},
			},
		},
		"options": map[string]any{
			"temperature": 0.3,
			"num_predict": 64,
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	req := httptest.NewRequest(http.MethodPost, "/api/generate", bytes.NewReader(body))
	req.Header.Set("Authorization", "Bearer test-key")
	rec := httptest.NewRecorder()

	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", rec.Code, rec.Body.String())
	}

	var result struct {
		Model         string `json:"model"`
		Response      string `json:"response"`
		Done          bool   `json:"done"`
		TotalDuration int64  `json:"total_duration"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result.Model != "groq:openai/gpt-oss-20b" {
		t.Fatalf("expected response model override, got %q", result.Model)
	}
	if result.Response != "ok" || !result.Done || result.TotalDuration <= 0 {
		t.Fatalf("unexpected response: %#v", result)
	}
	if provider.model != "groq:openai/gpt-oss-20b" {
		t.Fatalf("expected provider model override, got %q", provider.model)
	}
	if provider.request.System != "answer briefly" {
		t.Fatalf("expected system to be forwarded, got %q", provider.request.System)
	}
	if provider.request.Temperature == nil || *provider.request.Temperature != 0.3 {
		t.Fatalf("expected temperature to be forwarded, got %#v", provider.request.Temperature)
	}
	if provider.request.MaxOutputTokens == nil || *provider.request.MaxOutputTokens != 64 {
		t.Fatalf("expected num_predict to be forwarded, got %#v", provider.request.MaxOutputTokens)
	}
	if len(provider.request.ResponseSchema) == 0 {
		t.Fatal("expected response schema to be forwarded")
	}
}

func TestOllamaGenerateRejectsStreaming(t *testing.T) {
	service, err := application.NewAIService(fakeProvider{}, []string{"gemini:gemini-3.6-flash"})
	if err != nil {
		t.Fatal(err)
	}
	server := NewServer(service, slog.New(slog.DiscardHandler), []string{"test-key"}).Routes()

	body, err := json.Marshal(map[string]any{
		"model":  "gemini:gemini-3.6-flash",
		"prompt": "hello",
		"stream": true,
	})
	if err != nil {
		t.Fatal(err)
	}

	req := httptest.NewRequest(http.MethodPost, "/api/generate", bytes.NewReader(body))
	req.Header.Set("Authorization", "Bearer test-key")
	rec := httptest.NewRecorder()

	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected status 400, got %d: %s", rec.Code, rec.Body.String())
	}
}

func TestOllamaGenerateJSONFormat(t *testing.T) {
	provider := &capturingProvider{}
	service, err := application.NewAIService(provider, []string{"gemini:gemini-3.6-flash"})
	if err != nil {
		t.Fatal(err)
	}
	server := NewServer(service, slog.New(slog.DiscardHandler), []string{"test-key"}).Routes()

	body, err := json.Marshal(map[string]any{
		"prompt": "hello",
		"format": "json",
	})
	if err != nil {
		t.Fatal(err)
	}

	req := httptest.NewRequest(http.MethodPost, "/api/generate", bytes.NewReader(body))
	req.Header.Set("Authorization", "Bearer test-key")
	rec := httptest.NewRecorder()

	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", rec.Code, rec.Body.String())
	}
	if string(provider.request.ResponseSchema) != `{"type":"object"}` {
		t.Fatalf("unexpected JSON format schema: %s", provider.request.ResponseSchema)
	}
}

func TestChatCompletionsEndpoint(t *testing.T) {
	provider := &capturingProvider{}
	service, err := application.NewAIService(provider, []string{"gemini:gemini-3.6-flash"})
	if err != nil {
		t.Fatal(err)
	}
	server := NewServer(service, slog.New(slog.DiscardHandler), []string{"test-key"}).Routes()

	body, err := json.Marshal(map[string]any{
		"model": "groq:openai/gpt-oss-20b",
		"messages": []map[string]any{
			{"role": "system", "content": "answer briefly"},
			{"role": "user", "content": "hello"},
		},
		"temperature": 0.2,
		"max_tokens":  64,
	})
	if err != nil {
		t.Fatal(err)
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Authorization", "Bearer test-key")
	rec := httptest.NewRecorder()

	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", rec.Code, rec.Body.String())
	}

	var result struct {
		Object  string `json:"object"`
		Model   string `json:"model"`
		Choices []struct {
			Message struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result.Object != "chat.completion" || result.Model != "groq:openai/gpt-oss-20b" {
		t.Fatalf("unexpected completion response: %#v", result)
	}
	if len(result.Choices) != 1 || result.Choices[0].Message.Role != "assistant" || result.Choices[0].Message.Content != "ok" {
		t.Fatalf("unexpected choices: %#v", result.Choices)
	}
	if provider.model != "groq:openai/gpt-oss-20b" {
		t.Fatalf("expected provider model override, got %q", provider.model)
	}
	if len(provider.chatRequest.Messages) != 2 {
		t.Fatalf("expected messages to be forwarded, got %#v", provider.chatRequest.Messages)
	}
	if provider.chatRequest.Messages[0].Role != "system" || provider.chatRequest.Messages[1].Role != "user" {
		t.Fatalf("unexpected messages: %#v", provider.chatRequest.Messages)
	}
	if provider.chatRequest.Temperature == nil || *provider.chatRequest.Temperature != 0.2 {
		t.Fatalf("expected temperature to be forwarded, got %#v", provider.chatRequest.Temperature)
	}
	if provider.chatRequest.MaxOutputTokens == nil || *provider.chatRequest.MaxOutputTokens != 64 {
		t.Fatalf("expected max_tokens to be forwarded, got %#v", provider.chatRequest.MaxOutputTokens)
	}
}

func TestChatCompletionsEndpointAcceptsResponseFormat(t *testing.T) {
	provider := &capturingProvider{}
	service, err := application.NewAIService(provider, []string{"gemini:gemini-3.6-flash"})
	if err != nil {
		t.Fatal(err)
	}
	server := NewServer(service, slog.New(slog.DiscardHandler), []string{"test-key"}).Routes()

	body, err := json.Marshal(map[string]any{
		"messages": []map[string]any{
			{"role": "user", "content": "hello"},
		},
		"response_format": map[string]any{
			"type": "json_schema",
			"json_schema": map[string]any{
				"name":   "response",
				"strict": true,
				"schema": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"answer": map[string]any{"type": "string"},
					},
					"required": []string{"answer"},
				},
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Authorization", "Bearer test-key")
	rec := httptest.NewRecorder()

	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", rec.Code, rec.Body.String())
	}
	if len(provider.chatRequest.ResponseSchema) == 0 {
		t.Fatal("expected response schema to be forwarded")
	}
}

func TestChatCompletionsEndpointStreamsSingleChunk(t *testing.T) {
	service, err := application.NewAIService(fakeProvider{}, []string{"gemini:gemini-3.6-flash"})
	if err != nil {
		t.Fatal(err)
	}
	server := NewServer(service, slog.New(slog.DiscardHandler), []string{"test-key"}).Routes()

	body, err := json.Marshal(map[string]any{
		"stream": true,
		"messages": []map[string]any{
			{"role": "user", "content": "hello"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Authorization", "Bearer test-key")
	rec := httptest.NewRecorder()

	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", rec.Code, rec.Body.String())
	}
	if !strings.HasPrefix(rec.Header().Get("Content-Type"), "text/event-stream") {
		t.Fatalf("expected event stream content type, got %q", rec.Header().Get("Content-Type"))
	}
	if !strings.Contains(rec.Body.String(), "data: [DONE]") {
		t.Fatalf("expected stream done marker, got %s", rec.Body.String())
	}
}

func TestChatCompletionsEndpointUsesToolFallbacks(t *testing.T) {
	provider := &capturingProvider{}
	service, err := application.NewAIService(provider, []string{"gemini:gemini-3.6-flash"}, []string{"groq:openai/gpt-oss-120b"})
	if err != nil {
		t.Fatal(err)
	}
	server := NewServer(service, slog.New(slog.DiscardHandler), []string{"test-key"}).Routes()

	body, err := json.Marshal(map[string]any{
		"messages": []map[string]any{
			{"role": "user", "content": "hello"},
		},
		"tools": []map[string]any{
			{
				"type": "function",
				"function": map[string]any{
					"name":        "get_status",
					"description": "Get status",
					"parameters":  map[string]any{"type": "object"},
				},
			},
		},
		"tool_choice": "auto",
	})
	if err != nil {
		t.Fatal(err)
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Authorization", "Bearer test-key")
	rec := httptest.NewRecorder()

	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", rec.Code, rec.Body.String())
	}
	if provider.model != "groq:openai/gpt-oss-120b" {
		t.Fatalf("expected tool fallback model, got %q", provider.model)
	}
	if len(provider.chatRequest.Tools) == 0 {
		t.Fatal("expected tools to be forwarded")
	}
}

func TestChatCompletionsEndpointReturnsToolCalls(t *testing.T) {
	toolCalls := json.RawMessage(`[{"id":"call_1","type":"function","function":{"name":"get_status","arguments":"{}"}}]`)
	provider := &capturingProvider{
		chatResponse: domain.ChatResponse{
			Model: "gemini:gemini-3.6-flash",
			Message: domain.ChatMessage{
				Role:      "assistant",
				Content:   json.RawMessage("null"),
				ToolCalls: toolCalls,
			},
			FinishReason: "tool_calls",
		},
	}
	service, err := application.NewAIService(provider, []string{"gemini:gemini-3.6-flash"}, []string{"gemini:gemini-3.6-flash"})
	if err != nil {
		t.Fatal(err)
	}
	server := NewServer(service, slog.New(slog.DiscardHandler), []string{"test-key"}).Routes()

	body, err := json.Marshal(map[string]any{
		"messages": []map[string]any{
			{"role": "user", "content": "hello"},
		},
		"tools": []map[string]any{
			{
				"type": "function",
				"function": map[string]any{
					"name":       "get_status",
					"parameters": map[string]any{"type": "object"},
				},
			},
		},
		"tool_choice": "required",
	})
	if err != nil {
		t.Fatal(err)
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Authorization", "Bearer test-key")
	rec := httptest.NewRecorder()

	server.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", rec.Code, rec.Body.String())
	}

	var result struct {
		Choices []struct {
			Message struct {
				Content   any             `json:"content"`
				ToolCalls json.RawMessage `json:"tool_calls"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if len(result.Choices) != 1 || result.Choices[0].FinishReason != "tool_calls" {
		t.Fatalf("unexpected response: %#v", result)
	}
	if string(result.Choices[0].Message.ToolCalls) != string(toolCalls) {
		t.Fatalf("unexpected tool calls: %s", result.Choices[0].Message.ToolCalls)
	}
}

type fakeProvider struct{}

func (fakeProvider) Generate(_ context.Context, model string, _ domain.GenerateRequest) (domain.GenerateResponse, error) {
	return domain.GenerateResponse{Model: model, Text: "ok"}, nil
}

func (fakeProvider) Chat(_ context.Context, model string, _ domain.ChatRequest) (domain.ChatResponse, error) {
	return domain.ChatResponse{
		Model: model,
		Message: domain.ChatMessage{
			Role:    "assistant",
			Content: json.RawMessage(`"ok"`),
		},
		FinishReason: "stop",
	}, nil
}

type capturingProvider struct {
	model        string
	request      domain.GenerateRequest
	chatRequest  domain.ChatRequest
	chatResponse domain.ChatResponse
}

func (p *capturingProvider) Generate(_ context.Context, model string, req domain.GenerateRequest) (domain.GenerateResponse, error) {
	p.model = model
	p.request = req
	return domain.GenerateResponse{Model: model, Text: "ok"}, nil
}

func (p *capturingProvider) Chat(_ context.Context, model string, req domain.ChatRequest) (domain.ChatResponse, error) {
	p.model = model
	p.chatRequest = req
	if strings.TrimSpace(p.chatResponse.Model) != "" {
		return p.chatResponse, nil
	}
	return domain.ChatResponse{
		Model: model,
		Message: domain.ChatMessage{
			Role:    "assistant",
			Content: json.RawMessage(`"ok"`),
		},
		FinishReason: "stop",
	}, nil
}
