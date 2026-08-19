package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/frachea/ai-gateway/internal/domain"
)

func TestOpenRouterClientRequiresStructuredOutputParameters(t *testing.T) {
	var payload map[string]any

	client := NewOpenRouterClient("test-key", "https://openrouter.test", time.Second)
	client.httpClient = &http.Client{
		Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			if r.URL.Path != "/chat/completions" {
				t.Fatalf("unexpected path: %s", r.URL.Path)
			}
			if r.Header.Get("Authorization") != "Bearer test-key" {
				t.Fatalf("unexpected authorization header: %s", r.Header.Get("Authorization"))
			}
			if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
				t.Fatal(err)
			}
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body:       ioNopCloser{strings.NewReader(`{"choices":[{"message":{"content":"{\"answer\":\"ok\"}"}}]}`)},
			}, nil
		}),
	}

	_, err := client.Generate(context.Background(), "openrouter/free", domain.GenerateRequest{
		Prompt:         "hello",
		ResponseSchema: json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}`),
	})
	if err != nil {
		t.Fatal(err)
	}

	provider, ok := payload["provider"].(map[string]any)
	if !ok {
		t.Fatalf("expected provider options, got %#v", payload["provider"])
	}
	if provider["require_parameters"] != true {
		t.Fatalf("expected require_parameters true, got %#v", provider["require_parameters"])
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (fn roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) {
	return fn(r)
}

type ioNopCloser struct {
	*strings.Reader
}

func (c ioNopCloser) Close() error {
	return nil
}
