package gemini

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/frachea/ai-gateway/internal/domain"
)

func TestFunctionDeclarationsFromTools(t *testing.T) {
	tools := json.RawMessage(`[{"type":"function","function":{"name":"get_status","description":"Get status","parameters":{"type":"object","additionalProperties":false}}}]`)

	declarations, err := functionDeclarationsFromTools(tools)
	if err != nil {
		t.Fatal(err)
	}

	if len(declarations) != 1 || declarations[0].Name != "get_status" {
		t.Fatalf("unexpected declarations: %#v", declarations)
	}
	if string(declarations[0].Parameters) != `{"type":"object"}` {
		t.Fatalf("unexpected parameters: %s", declarations[0].Parameters)
	}
}

func TestFunctionDeclarationsFromToolsStripsSchemaKeyword(t *testing.T) {
	tools := json.RawMessage(`[{"type":"function","function":{"name":"pods_list","parameters":{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","properties":{"namespace":{"type":"string"}}}}}]`)

	declarations, err := functionDeclarationsFromTools(tools)
	if err != nil {
		t.Fatal(err)
	}

	if len(declarations) != 1 {
		t.Fatalf("unexpected declarations: %#v", declarations)
	}
	if string(declarations[0].Parameters) != `{"properties":{"namespace":{"type":"string"}},"type":"object"}` {
		t.Fatalf("expected $schema to be stripped, got %s", declarations[0].Parameters)
	}
}

func TestFunctionDeclarationsFromToolsConvertsConstToEnum(t *testing.T) {
	tools := json.RawMessage(`[{"type":"function","function":{"name":"set_mode","parameters":{"type":"object","properties":{"mode":{"const":"read-only"}}}}}]`)

	declarations, err := functionDeclarationsFromTools(tools)
	if err != nil {
		t.Fatal(err)
	}

	if len(declarations) != 1 {
		t.Fatalf("unexpected declarations: %#v", declarations)
	}
	if string(declarations[0].Parameters) != `{"properties":{"mode":{"enum":["read-only"]}},"type":"object"}` {
		t.Fatalf("expected const to be converted to enum, got %s", declarations[0].Parameters)
	}
}

func TestFunctionDeclarationsFromToolsKeepsOneOfAsAnyOf(t *testing.T) {
	tools := json.RawMessage(`[{"type":"function","function":{"name":"apply","parameters":{
		"type":"object",
		"properties":{
			"operations":{"type":"array","items":{"oneOf":[
				{"type":"object","properties":{"op":{"type":"string","const":"add"},"name":{"type":"string"}},"required":["op","name"]},
				{"type":"object","properties":{"op":{"type":"string","const":"remove"},"id":{"type":"string"}},"required":["op","id"]}
			]}}
		}
	}}}]`)

	declarations, err := functionDeclarationsFromTools(tools)
	if err != nil {
		t.Fatal(err)
	}

	want := `{"properties":{"operations":{"items":{"anyOf":[` +
		`{"properties":{"name":{"type":"string"},"op":{"enum":["add"],"type":"string"}},"required":["op","name"],"type":"object"},` +
		`{"properties":{"id":{"type":"string"},"op":{"enum":["remove"],"type":"string"}},"required":["op","id"],"type":"object"}` +
		`]},"type":"array"}},"type":"object"}`
	if string(declarations[0].Parameters) != want {
		t.Fatalf("expected oneOf to survive as anyOf, got %s", declarations[0].Parameters)
	}
}

func TestFunctionDeclarationsFromToolsStripsUnknownJSONSchemaKeywords(t *testing.T) {
	tools := json.RawMessage(`[{"type":"function","function":{"name":"complex_tool","parameters":{
		"type":"object",
		"patternProperties":{"^x-":{"type":"string"}},
		"properties":{
			"count":{"type":"integer","exclusiveMinimum":0,"minimum":1}
		}
	}}}]`)

	declarations, err := functionDeclarationsFromTools(tools)
	if err != nil {
		t.Fatal(err)
	}

	if len(declarations) != 1 {
		t.Fatalf("unexpected declarations: %#v", declarations)
	}
	if string(declarations[0].Parameters) != `{"properties":{"count":{"type":"integer"}},"type":"object"}` {
		t.Fatalf("expected unrecognized keywords to be stripped, got %s", declarations[0].Parameters)
	}
}

func TestGenerateContentResponseReturnsToolCalls(t *testing.T) {
	var response generateContentResponse
	body := `{
		"candidates": [{
			"finishReason": "STOP",
			"content": {
				"parts": [
					{
						"functionCall": {"name": "get_status", "args": {"namespace": "ai"}},
						"thoughtSignature": "sig-abc"
					}
				]
			}
		}]
	}`
	if err := json.Unmarshal([]byte(body), &response); err != nil {
		t.Fatal(err)
	}

	message, finishReason, err := response.ChatMessage()
	if err != nil {
		t.Fatal(err)
	}

	if finishReason != "tool_calls" {
		t.Fatalf("expected tool_calls finish reason, got %q", finishReason)
	}
	if len(message.ToolCalls) == 0 {
		t.Fatal("expected tool calls")
	}

	var calls []openAIToolCall
	if err := json.Unmarshal(message.ToolCalls, &calls); err != nil {
		t.Fatal(err)
	}
	if len(calls) != 1 || calls[0].ThoughtSignature != "sig-abc" {
		t.Fatalf("expected thought signature to be captured, got %#v", calls)
	}
}

func TestContentsFromMessagesMapsToolResults(t *testing.T) {
	contents, _, err := contentsFromMessages([]domain.ChatMessage{
		{Role: "user", Content: json.RawMessage(`"status?"`)},
		{
			Role:      "assistant",
			Content:   json.RawMessage("null"),
			ToolCalls: json.RawMessage(`[{"id":"call_1","type":"function","function":{"name":"get_status","arguments":"{\"namespace\":\"ai\"}"}}]`),
		},
		{Role: "tool", ToolCallID: "call_1", Content: json.RawMessage(`"{\"ready\":true}"`)},
	})
	if err != nil {
		t.Fatal(err)
	}

	if len(contents) != 3 {
		t.Fatalf("expected 3 contents, got %d", len(contents))
	}
	last := contents[2]
	if last.Role != "user" || len(last.Parts) != 1 || last.Parts[0].FunctionResponse == nil {
		t.Fatalf("unexpected tool result content: %#v", last)
	}
	if last.Parts[0].FunctionResponse.Name != "get_status" {
		t.Fatalf("unexpected function response name: %q", last.Parts[0].FunctionResponse.Name)
	}
}

func TestContentsFromMessagesReplaysThoughtSignature(t *testing.T) {
	contents, _, err := contentsFromMessages([]domain.ChatMessage{
		{Role: "user", Content: json.RawMessage(`"status?"`)},
		{
			Role:      "assistant",
			Content:   json.RawMessage("null"),
			ToolCalls: json.RawMessage(`[{"id":"call_1","type":"function","function":{"name":"get_status","arguments":"{}"},"thought_signature":"sig-abc"},{"id":"call_2","type":"function","function":{"name":"get_other","arguments":"{}"}}]`),
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	if len(contents) != 2 {
		t.Fatalf("expected 2 contents, got %d", len(contents))
	}
	model := contents[1]
	if len(model.Parts) != 2 {
		t.Fatalf("expected 2 function call parts, got %#v", model.Parts)
	}
	if model.Parts[0].ThoughtSignature != "sig-abc" {
		t.Fatalf("expected real thought signature to be replayed, got %q", model.Parts[0].ThoughtSignature)
	}
	if model.Parts[1].ThoughtSignature != "skip_thought_signature_validator" {
		t.Fatalf("expected placeholder thought signature for unsigned call, got %q", model.Parts[1].ThoughtSignature)
	}
}

func TestClassifyGeminiErrorTreatsTokenLimitAsFallbackable(t *testing.T) {
	err := classifyGeminiError(http.StatusBadRequest, []byte(`{"error":{"code":400,"message":"Input exceeds the maximum token limit","status":"INVALID_ARGUMENT"}}`))

	if domain.Kind(err) != domain.ErrorKindModelUnavailable {
		t.Fatalf("expected model unavailable error, got %q", domain.Kind(err))
	}
}

func TestClientChatRetriesWithoutArrayBoundsWhenSchemaIsTooComplex(t *testing.T) {
	var bodies []string
	client := NewClient("test-key", "https://gemini.test", time.Second)
	client.httpClient = &http.Client{
		Transport: geminiRoundTrip(func(r *http.Request) (*http.Response, error) {
			raw, _ := io.ReadAll(r.Body)
			bodies = append(bodies, string(raw))
			if strings.Contains(string(raw), "maxItems") {
				return geminiResponse(http.StatusBadRequest, `{"error":{"code":400,"status":"INVALID_ARGUMENT","message":"The specified schema produces a constraint that has too many states for serving."}}`), nil
			}
			return geminiResponse(http.StatusOK, `{"candidates":[{"content":{"parts":[{"functionCall":{"name":"apply","args":{"items":["a"]}}}]},"finishReason":"STOP"}]}`), nil
		}),
	}

	response, err := client.Chat(context.Background(), "gemini-3.5-flash-lite", domain.ChatRequest{
		Messages: []domain.ChatMessage{{Role: "user", Content: json.RawMessage(`"go"`)}},
		Tools:    json.RawMessage(`[{"type":"function","function":{"name":"apply","parameters":{"type":"object","properties":{"items":{"type":"array","minItems":1,"maxItems":24,"items":{"type":"string"}}}}}}]`),
	})
	if err != nil {
		t.Fatal(err)
	}

	if len(bodies) != 2 {
		t.Fatalf("expected one retry, got %d requests", len(bodies))
	}
	if strings.Contains(bodies[1], "minItems") || strings.Contains(bodies[1], "maxItems") {
		t.Fatalf("expected the retry to drop array bounds, got %s", bodies[1])
	}
	if response.FinishReason != "tool_calls" {
		t.Fatalf("expected the retried call to succeed, got %#v", response)
	}
}

func TestClassifyGeminiErrorTreatsTooComplexSchemaAsFallbackable(t *testing.T) {
	err := classifyGeminiError(http.StatusBadRequest, []byte(`{"error":{"code":400,"status":"INVALID_ARGUMENT","message":"The specified schema produces a constraint that has too many states for serving."}}`))

	if domain.Kind(err) != domain.ErrorKindModelUnavailable {
		t.Fatalf("expected model unavailable error, got %q", domain.Kind(err))
	}
}

type geminiRoundTrip func(*http.Request) (*http.Response, error)

func (fn geminiRoundTrip) RoundTrip(r *http.Request) (*http.Response, error) {
	return fn(r)
}

func geminiResponse(status int, body string) *http.Response {
	return &http.Response{
		StatusCode: status,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(strings.NewReader(body)),
	}
}
