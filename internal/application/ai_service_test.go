package application

import (
	"context"
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
	service, err := NewAIService(fakeProvider{}, []string{"gemini:gemini-3.6-flash", "groq:openai/gpt-oss-20b"})
	if err != nil {
		t.Fatal(err)
	}

	models := service.Models()
	if len(models) != 2 {
		t.Fatalf("expected 2 models, got %d", len(models))
	}
	if models[0].Provider != "gemini" || models[0].Model != "gemini-3.6-flash" || models[0].Order != 1 {
		t.Fatalf("unexpected first model: %#v", models[0])
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

type fakeProvider struct {
	responses map[string]domain.GenerateResponse
	errors    map[string]error
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
