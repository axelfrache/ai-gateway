package httpapi

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
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

type fakeProvider struct{}

func (fakeProvider) Generate(_ context.Context, model string, _ domain.GenerateRequest) (domain.GenerateResponse, error) {
	return domain.GenerateResponse{Model: model, Text: "ok"}, nil
}
