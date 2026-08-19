package mcpsmoke

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestServerListsAndCallsTools(t *testing.T) {
	server := NewServer().Routes()

	initialize := rpc(t, server, `{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25"}}`)
	if initialize["error"] != nil {
		t.Fatalf("unexpected initialize error: %#v", initialize["error"])
	}

	tools := rpc(t, server, `{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}`)
	result := tools["result"].(map[string]any)
	list := result["tools"].([]any)
	if len(list) != 2 {
		t.Fatalf("expected 2 tools, got %d", len(list))
	}

	status := rpc(t, server, `{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"status","arguments":{}}}`)
	statusResult := status["result"].(map[string]any)
	if statusResult["isError"] == true {
		t.Fatalf("unexpected status error: %#v", statusResult)
	}
	structured := statusResult["structuredContent"].(map[string]any)
	if structured["ok"] != true || structured["name"] != "mcp-smoke" {
		t.Fatalf("unexpected structured content: %#v", structured)
	}

	echo := rpc(t, server, `{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"echo","arguments":{"message":"hello","payload":{"x":1}}}}`)
	echoResult := echo["result"].(map[string]any)
	echoStructured := echoResult["structuredContent"].(map[string]any)
	if echoStructured["message"] != "hello" {
		t.Fatalf("unexpected echo result: %#v", echoStructured)
	}
}

func TestInitializedNotificationReturnsAccepted(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/mcp", bytes.NewBufferString(`{"jsonrpc":"2.0","method":"notifications/initialized"}`))
	res := httptest.NewRecorder()

	NewServer().Routes().ServeHTTP(res, req)

	if res.Code != http.StatusAccepted {
		t.Fatalf("expected 202, got %d", res.Code)
	}
}

func rpc(t *testing.T, handler http.Handler, payload string) map[string]any {
	t.Helper()

	req := httptest.NewRequest(http.MethodPost, "/mcp", bytes.NewBufferString(payload))
	res := httptest.NewRecorder()
	handler.ServeHTTP(res, req)

	if res.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", res.Code)
	}

	var response map[string]any
	if err := json.NewDecoder(res.Body).Decode(&response); err != nil {
		t.Fatal(err)
	}
	return response
}
