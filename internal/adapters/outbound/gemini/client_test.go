package gemini

import (
	"encoding/json"
	"net/http"
	"testing"

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
