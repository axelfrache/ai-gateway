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

func TestClientChatForwardsTools(t *testing.T) {
	var payload map[string]any

	client := NewClient("Groq", "test-key", "https://groq.test", "max_completion_tokens", time.Second)
	client.httpClient = &http.Client{
		Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
				t.Fatal(err)
			}
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body:       ioNopCloser{strings.NewReader(`{"choices":[{"finish_reason":"tool_calls","message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_status","arguments":"{}"}}]}}]}`)},
			}, nil
		}),
	}

	tools := json.RawMessage(`[{"type":"function","function":{"name":"get_status","parameters":{"type":"object"}}}]`)
	response, err := client.Chat(context.Background(), "openai/gpt-oss-120b", domain.ChatRequest{
		Messages: []domain.ChatMessage{
			{Role: "user", Content: json.RawMessage(`"hello"`)},
		},
		Tools:      tools,
		ToolChoice: json.RawMessage(`"required"`),
	})
	if err != nil {
		t.Fatal(err)
	}

	if payload["tool_choice"] != "required" {
		t.Fatalf("expected tool_choice to be forwarded, got %#v", payload["tool_choice"])
	}
	if _, ok := payload["tools"].([]any); !ok {
		t.Fatalf("expected tools to be forwarded, got %#v", payload["tools"])
	}
	if string(response.Message.ToolCalls) == "" || response.FinishReason != "tool_calls" {
		t.Fatalf("unexpected tool response: %#v", response)
	}
}

func TestMistralClientMapsRequiredToolChoiceToAny(t *testing.T) {
	var payload map[string]any

	client := NewClient("Mistral", "test-key", "https://mistral.test", "max_tokens", time.Second)
	client.httpClient = &http.Client{
		Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
				t.Fatal(err)
			}
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body:       ioNopCloser{strings.NewReader(`{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`)},
			}, nil
		}),
	}

	_, err := client.Chat(context.Background(), "mistral-small-latest", domain.ChatRequest{
		Messages: []domain.ChatMessage{
			{Role: "user", Content: json.RawMessage(`"hello"`)},
		},
		Tools:      json.RawMessage(`[{"type":"function","function":{"name":"get_status","parameters":{"type":"object"}}}]`),
		ToolChoice: json.RawMessage(`"required"`),
	})
	if err != nil {
		t.Fatal(err)
	}

	if payload["tool_choice"] != "any" {
		t.Fatalf("expected Mistral tool_choice any, got %#v", payload["tool_choice"])
	}
}

func TestClassifyErrorTreatsTokenLimitAsFallbackable(t *testing.T) {
	err := classifyError("Groq", http.StatusBadRequest, []byte(`{"error":{"message":"This model's maximum context length was exceeded"}}`))

	if domain.Kind(err) != domain.ErrorKindModelUnavailable {
		t.Fatalf("expected model unavailable error, got %q", domain.Kind(err))
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
