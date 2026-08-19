package application

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/frachea/ai-gateway/internal/domain"
)

func TestGenerateUsesFirstSuccessfulModel(t *testing.T) {
	provider := fakeProvider{
		responses: map[string]domain.GenerateResponse{
			"gemini-3.7-flash": {Model: "gemini-3.7-flash", Text: "ok"},
		},
	}
	service, err := NewAIService(provider, []string{"gemini-3.7-flash", "gemini-3.6-flash"})
	if err != nil {
		t.Fatal(err)
	}

	result, err := service.Generate(context.Background(), domain.GenerateRequest{Prompt: "hello"})
	if err != nil {
		t.Fatal(err)
	}

	if result.Model != "gemini-3.7-flash" {
		t.Fatalf("expected first model, got %q", result.Model)
	}
	if result.FallbackUsed {
		t.Fatal("expected fallback_used=false")
	}
	if len(result.Attempts) != 1 {
		t.Fatalf("expected 1 attempt, got %d", len(result.Attempts))
	}
}

func TestGenerateFallsBackOnRetryableError(t *testing.T) {
	provider := fakeProvider{
		errors: map[string]error{
			"gemini-3.7-flash": domain.NewError(domain.ErrorKindRateLimited, 429, "quota exceeded", nil),
		},
		responses: map[string]domain.GenerateResponse{
			"gemini-3.6-flash": {Model: "gemini-3.6-flash", Text: "fallback ok"},
		},
	}
	service, err := NewAIService(provider, []string{"gemini-3.7-flash", "gemini-3.6-flash"})
	if err != nil {
		t.Fatal(err)
	}

	result, err := service.Generate(context.Background(), domain.GenerateRequest{Prompt: "hello"})
	if err != nil {
		t.Fatal(err)
	}

	if result.Model != "gemini-3.6-flash" {
		t.Fatalf("expected fallback model, got %q", result.Model)
	}
	if !result.FallbackUsed {
		t.Fatal("expected fallback_used=true")
	}
	if len(result.Attempts) != 2 {
		t.Fatalf("expected 2 attempts, got %d", len(result.Attempts))
	}
}

func TestGenerateStopsOnNonRetryableError(t *testing.T) {
	provider := fakeProvider{
		errors: map[string]error{
			"gemini-3.7-flash": domain.NewError(domain.ErrorKindInvalidRequest, 400, "bad prompt", nil),
		},
		responses: map[string]domain.GenerateResponse{
			"gemini-3.6-flash": {Model: "gemini-3.6-flash", Text: "should not be used"},
		},
	}
	service, err := NewAIService(provider, []string{"gemini-3.7-flash", "gemini-3.6-flash"})
	if err != nil {
		t.Fatal(err)
	}

	result, err := service.Generate(context.Background(), domain.GenerateRequest{Prompt: "hello"})
	if err == nil {
		t.Fatal("expected error")
	}
	if domain.Kind(err) != domain.ErrorKindInvalidRequest {
		t.Fatalf("expected invalid request error, got %q", domain.Kind(err))
	}
	if len(result.Attempts) != 1 {
		t.Fatalf("expected 1 attempt, got %d", len(result.Attempts))
	}
}

func TestGenerateUsesRequestedModel(t *testing.T) {
	provider := fakeProvider{
		responses: map[string]domain.GenerateResponse{
			"groq:openai/gpt-oss-20b": {Model: "groq:openai/gpt-oss-20b", Text: "ok"},
		},
	}
	service, err := NewAIService(provider, []string{"gemini:gemini-3.6-flash"})
	if err != nil {
		t.Fatal(err)
	}

	result, err := service.Generate(context.Background(), domain.GenerateRequest{
		Prompt: "hello",
		Model:  "groq:openai/gpt-oss-20b",
	})
	if err != nil {
		t.Fatal(err)
	}

	if result.Model != "groq:openai/gpt-oss-20b" {
		t.Fatalf("expected requested model, got %q", result.Model)
	}
}

func TestGenerateRejectsModelAndModelsTogether(t *testing.T) {
	service, err := NewAIService(fakeProvider{}, []string{"gemini:gemini-3.6-flash"})
	if err != nil {
		t.Fatal(err)
	}

	_, err = service.Generate(context.Background(), domain.GenerateRequest{
		Prompt: "hello",
		Model:  "gemini:gemini-3.6-flash",
		Models: []string{"groq:openai/gpt-oss-20b"},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if domain.Kind(err) != domain.ErrorKindInvalidRequest {
		t.Fatalf("expected invalid request error, got %q", domain.Kind(err))
	}
}

func TestModelsReturnsConfiguredModels(t *testing.T) {
	service, err := NewAIService(
		fakeProvider{},
		[]string{"gemini:gemini-3.6-flash"},
		[]string{"gemini:gemini-3.6-flash", "groq:openai/gpt-oss-20b"},
	)
	if err != nil {
		t.Fatal(err)
	}

	models := service.Models()
	if len(models) != 5 {
		t.Fatalf("expected 3 aliases + 2 real models, got %d: %#v", len(models), models)
	}

	byName := map[string]domain.ModelInfo{}
	for _, model := range models {
		byName[model.Name] = model
	}

	if !byName[AliasAuto].SupportsJSON || byName[AliasAuto].SupportsTools {
		t.Fatalf("unexpected auto alias metadata: %#v", byName[AliasAuto])
	}
	if !byName[AliasTools].SupportsTools || byName[AliasTools].SupportsJSON {
		t.Fatalf("unexpected tools alias metadata: %#v", byName[AliasTools])
	}
	if !byName[AliasJSON].SupportsJSON || byName[AliasJSON].SupportsTools {
		t.Fatalf("unexpected json alias metadata: %#v", byName[AliasJSON])
	}

	real := byName["gemini:gemini-3.6-flash"]
	if real.Provider != "gemini" || real.Model != "gemini-3.6-flash" {
		t.Fatalf("unexpected real model: %#v", real)
	}
	if !real.SupportsTools {
		t.Fatalf("expected tool support metadata, got %#v", real)
	}
}

func TestGenerateResolvesJSONAlias(t *testing.T) {
	provider := fakeProvider{
		responses: map[string]domain.GenerateResponse{
			"mistral:mistral-small-latest": {Model: "mistral:mistral-small-latest", Text: "ok"},
		},
	}
	service, err := NewAIService(
		provider,
		[]string{"gemini:gemini-3.6-flash"},
		[]string{"groq:openai/gpt-oss-120b"},
		[]string{"mistral:mistral-small-latest"},
	)
	if err != nil {
		t.Fatal(err)
	}

	result, err := service.Generate(context.Background(), domain.GenerateRequest{
		Prompt: "hello",
		Model:  AliasJSON,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Model != "mistral:mistral-small-latest" {
		t.Fatalf("expected json alias model, got %q", result.Model)
	}
}

func TestChatResolvesToolsAlias(t *testing.T) {
	provider := fakeProvider{
		chatResponses: map[string]domain.ChatResponse{
			"groq:openai/gpt-oss-120b": {
				Model: "groq:openai/gpt-oss-120b",
				Message: domain.ChatMessage{
					Role:    "assistant",
					Content: []byte(`"ok"`),
				},
			},
		},
	}
	service, err := NewAIService(
		provider,
		[]string{"gemini:gemini-3.6-flash"},
		[]string{"groq:openai/gpt-oss-120b"},
	)
	if err != nil {
		t.Fatal(err)
	}

	result, err := service.Chat(context.Background(), domain.ChatRequest{
		Messages: []domain.ChatMessage{{Role: "user", Content: []byte(`"hello"`)}},
		Model:    AliasTools,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Model != "groq:openai/gpt-oss-120b" {
		t.Fatalf("expected tools alias model, got %q", result.Model)
	}
}

func TestCheckModelsExpandsAlias(t *testing.T) {
	provider := fakeProvider{
		errors: map[string]error{
			"groq:openai/gpt-oss-120b": domain.NewError(domain.ErrorKindTemporary, 500, "down", nil),
		},
	}
	service, err := NewAIService(
		provider,
		[]string{"gemini:gemini-3.6-flash"},
		[]string{"groq:openai/gpt-oss-120b"},
	)
	if err != nil {
		t.Fatal(err)
	}

	checks := service.CheckModels(context.Background(), []string{AliasTools})
	if len(checks) != 1 {
		t.Fatalf("expected 1 expanded check, got %d: %#v", len(checks), checks)
	}
	if checks[0].Name != "groq:openai/gpt-oss-120b" || checks[0].Status != "unavailable" {
		t.Fatalf("unexpected check result: %#v", checks[0])
	}
}

func TestGenerateRequiresPrompt(t *testing.T) {
	service, err := NewAIService(fakeProvider{}, []string{"gemini-3.7-flash"})
	if err != nil {
		t.Fatal(err)
	}

	_, err = service.Generate(context.Background(), domain.GenerateRequest{})
	if err == nil {
		t.Fatal("expected error")
	}
	if domain.Kind(err) != domain.ErrorKindInvalidRequest {
		t.Fatalf("expected invalid request error, got %q", domain.Kind(err))
	}
}

func TestChatUsesToolFallbacks(t *testing.T) {
	provider := fakeProvider{
		chatResponses: map[string]domain.ChatResponse{
			"groq:openai/gpt-oss-120b": {
				Model: "groq:openai/gpt-oss-120b",
				Message: domain.ChatMessage{
					Role:    "assistant",
					Content: []byte(`"ok"`),
				},
			},
		},
	}
	service, err := NewAIService(provider, []string{"gemini:gemini-3.6-flash"}, []string{"groq:openai/gpt-oss-120b"})
	if err != nil {
		t.Fatal(err)
	}

	result, err := service.Chat(context.Background(), domain.ChatRequest{
		Messages: []domain.ChatMessage{
			{Role: "user", Content: []byte(`"hello"`)},
		},
		Tools: []byte(`[{"type":"function","function":{"name":"get_status"}}]`),
	})
	if err != nil {
		t.Fatal(err)
	}

	if result.Model != "groq:openai/gpt-oss-120b" {
		t.Fatalf("expected tool fallback model, got %q", result.Model)
	}
}

func TestChatFallsBackOnRetryableError(t *testing.T) {
	provider := fakeProvider{
		chatErrors: map[string]error{
			"gemini:gemini-3.6-flash": domain.NewError(domain.ErrorKindRateLimited, 429, "quota exceeded", nil),
		},
		chatResponses: map[string]domain.ChatResponse{
			"groq:openai/gpt-oss-120b": {
				Model: "groq:openai/gpt-oss-120b",
				Message: domain.ChatMessage{
					Role:    "assistant",
					Content: []byte(`"ok"`),
				},
			},
		},
	}
	service, err := NewAIService(provider, []string{"gemini:gemini-3.6-flash", "groq:openai/gpt-oss-120b"})
	if err != nil {
		t.Fatal(err)
	}

	result, err := service.Chat(context.Background(), domain.ChatRequest{
		Messages: []domain.ChatMessage{
			{Role: "user", Content: []byte(`"hello"`)},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	if result.Model != "groq:openai/gpt-oss-120b" || !result.FallbackUsed {
		t.Fatalf("unexpected fallback result: %#v", result)
	}
	if len(result.Attempts) != 2 {
		t.Fatalf("expected 2 attempts, got %d", len(result.Attempts))
	}
}

func TestChatExecutesGatewayToolCalls(t *testing.T) {
	provider := &scriptedChatProvider{
		responses: []domain.ChatResponse{
			{
				Model: "gemini:gemini-3.6-flash",
				Message: domain.ChatMessage{
					Role:      "assistant",
					Content:   json.RawMessage("null"),
					ToolCalls: json.RawMessage(`[{"id":"call_1","type":"function","function":{"name":"kube__get_pods","arguments":"{\"namespace\":\"ai\"}"}}]`),
				},
				FinishReason: "tool_calls",
			},
			{
				Model: "gemini:gemini-3.6-flash",
				Message: domain.ChatMessage{
					Role:    "assistant",
					Content: json.RawMessage(`"Pods are ready"`),
				},
				FinishReason: "stop",
			},
		},
	}
	executor := &fakeToolExecutor{
		tools: []domain.ToolDefinition{
			{
				Name:        "kube__get_pods",
				Description: "List pods",
				Parameters:  json.RawMessage(`{"type":"object","properties":{"namespace":{"type":"string"}}}`),
			},
		},
		result: domain.ToolResult{
			ToolCallID: "call_1",
			Name:       "kube__get_pods",
			Content:    `{"pods":["ai-gateway"]}`,
		},
	}
	service, err := NewAIService(provider, []string{"gemini:gemini-3.6-flash"})
	if err != nil {
		t.Fatal(err)
	}
	service.SetToolExecutor(executor, 4)

	result, err := service.Chat(context.Background(), domain.ChatRequest{
		Messages: []domain.ChatMessage{
			{Role: "user", Content: json.RawMessage(`"list ai pods"`)},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	if result.FinishReason != "stop" || string(result.Message.Content) != `"Pods are ready"` {
		t.Fatalf("unexpected final response: %#v", result)
	}
	if provider.calls != 2 {
		t.Fatalf("expected 2 chat calls, got %d", provider.calls)
	}
	if executor.executedName != "kube__get_pods" || executor.executedArguments != `{"namespace":"ai"}` {
		t.Fatalf("unexpected tool execution: %q %q", executor.executedName, executor.executedArguments)
	}
	lastRequest := provider.requests[0]
	if len(lastRequest.Tools) == 0 {
		t.Fatal("expected gateway tools to be injected")
	}
	if len(provider.requests[1].Messages) != 3 {
		t.Fatalf("expected tool result appended before second chat call, got %d messages", len(provider.requests[1].Messages))
	}
}

func TestChatDoesNotPartiallyExecuteMixedToolCalls(t *testing.T) {
	provider := &scriptedChatProvider{
		responses: []domain.ChatResponse{
			{
				Model: "gemini:gemini-3.6-flash",
				Message: domain.ChatMessage{
					Role:      "assistant",
					Content:   json.RawMessage("null"),
					ToolCalls: json.RawMessage(`[{"id":"call_1","type":"function","function":{"name":"kube__get_pods","arguments":"{}"}},{"id":"call_2","type":"function","function":{"name":"caller_tool","arguments":"{}"}}]`),
				},
				FinishReason: "tool_calls",
			},
		},
	}
	executor := &fakeToolExecutor{
		tools: []domain.ToolDefinition{{Name: "kube__get_pods", Parameters: json.RawMessage(`{"type":"object"}`)}},
	}
	service, err := NewAIService(provider, []string{"gemini:gemini-3.6-flash"})
	if err != nil {
		t.Fatal(err)
	}
	service.SetToolExecutor(executor, 4)

	result, err := service.Chat(context.Background(), domain.ChatRequest{
		Messages: []domain.ChatMessage{{Role: "user", Content: json.RawMessage(`"run tools"`)}},
		Tools:    json.RawMessage(`[{"type":"function","function":{"name":"caller_tool","parameters":{"type":"object"}}}]`),
	})
	if err != nil {
		t.Fatal(err)
	}

	if provider.calls != 1 {
		t.Fatalf("expected one chat call, got %d", provider.calls)
	}
	if executor.executedName != "" {
		t.Fatalf("expected no partial execution, got %q", executor.executedName)
	}
	if len(result.Message.ToolCalls) == 0 {
		t.Fatal("expected tool calls to be returned")
	}
}

type fakeProvider struct {
	responses     map[string]domain.GenerateResponse
	errors        map[string]error
	chatResponses map[string]domain.ChatResponse
	chatErrors    map[string]error
}

func (p fakeProvider) Generate(_ context.Context, model string, _ domain.GenerateRequest) (domain.GenerateResponse, error) {
	if p.errors != nil {
		if err := p.errors[model]; err != nil {
			return domain.GenerateResponse{}, err
		}
	}
	if p.responses != nil {
		if response, ok := p.responses[model]; ok {
			return response, nil
		}
	}
	return domain.GenerateResponse{}, errors.New("unexpected model")
}

func (p fakeProvider) Chat(_ context.Context, model string, _ domain.ChatRequest) (domain.ChatResponse, error) {
	if p.chatErrors != nil {
		if err := p.chatErrors[model]; err != nil {
			return domain.ChatResponse{}, err
		}
	}
	if p.chatResponses != nil {
		if response, ok := p.chatResponses[model]; ok {
			return response, nil
		}
	}
	return domain.ChatResponse{}, errors.New("unexpected model")
}

type scriptedChatProvider struct {
	responses []domain.ChatResponse
	requests  []domain.ChatRequest
	calls     int
}

func (p *scriptedChatProvider) Generate(_ context.Context, _ string, _ domain.GenerateRequest) (domain.GenerateResponse, error) {
	return domain.GenerateResponse{}, errors.New("unexpected generate")
}

func (p *scriptedChatProvider) Chat(_ context.Context, _ string, req domain.ChatRequest) (domain.ChatResponse, error) {
	p.requests = append(p.requests, req)
	if p.calls >= len(p.responses) {
		return domain.ChatResponse{}, errors.New("unexpected chat call")
	}
	response := p.responses[p.calls]
	p.calls++
	return response, nil
}

type fakeToolExecutor struct {
	tools             []domain.ToolDefinition
	result            domain.ToolResult
	executedName      string
	executedArguments string
}

func (e *fakeToolExecutor) ListTools(_ context.Context) ([]domain.ToolDefinition, error) {
	return e.tools, nil
}

func (e *fakeToolExecutor) ExecuteTool(_ context.Context, toolCallID, name, arguments string) (domain.ToolResult, error) {
	e.executedName = name
	e.executedArguments = arguments
	result := e.result
	result.ToolCallID = toolCallID
	result.Name = name
	return result, nil
}
