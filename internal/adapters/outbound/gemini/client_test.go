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

func TestGenerateContentResponseReturnsToolCalls(t *testing.T) {
	response := generateContentResponse{
		Candidates: []struct {
			Content struct {
				Parts []struct {
					Text         string        `json:"text"`
					FunctionCall *functionCall `json:"functionCall"`
				} `json:"parts"`
			} `json:"content"`
			FinishReason string `json:"finishReason"`
		}{
			{
				FinishReason: "STOP",
				Content: struct {
					Parts []struct {
						Text         string        `json:"text"`
						FunctionCall *functionCall `json:"functionCall"`
					} `json:"parts"`
				}{
					Parts: []struct {
						Text         string        `json:"text"`
						FunctionCall *functionCall `json:"functionCall"`
					}{
						{FunctionCall: &functionCall{Name: "get_status", Args: json.RawMessage(`{"namespace":"ai"}`)}},
					},
				},
			},
		},
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

func TestClassifyGeminiErrorTreatsTokenLimitAsFallbackable(t *testing.T) {
	err := classifyGeminiError(http.StatusBadRequest, []byte(`{"error":{"code":400,"message":"Input exceeds the maximum token limit","status":"INVALID_ARGUMENT"}}`))

	if domain.Kind(err) != domain.ErrorKindModelUnavailable {
		t.Fatalf("expected model unavailable error, got %q", domain.Kind(err))
	}
}
