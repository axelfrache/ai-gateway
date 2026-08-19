package mcp

import (
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"
)

func TestRegistryListsAndExecutesTools(t *testing.T) {
	var methods []string
	var calledTool string
	var calledArguments map[string]any

	client := NewClient("kube", "http://mcp.test/mcp", "test-token", "2025-11-25", time.Second)
	client.httpClient = &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		if r.Header.Get("Accept") != "application/json, text/event-stream" {
			t.Fatalf("unexpected accept header: %q", r.Header.Get("Accept"))
		}
		if r.Header.Get("Authorization") != "Bearer test-token" {
			t.Fatalf("unexpected authorization header: %q", r.Header.Get("Authorization"))
		}

		var payload struct {
			ID     any             `json:"id"`
			Method string          `json:"method"`
			Params json.RawMessage `json:"params"`
		}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatal(err)
		}
		methods = append(methods, payload.Method)

		header := http.Header{"Content-Type": []string{"application/json"}}
		switch payload.Method {
		case "initialize":
			header.Set("Mcp-Session-Id", "session-1")
			return response(http.StatusOK, header, `{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-11-25","capabilities":{},"serverInfo":{"name":"kube","version":"test"}}}`), nil
		case "notifications/initialized":
			return response(http.StatusAccepted, header, ""), nil
		case "tools/list":
			if r.Header.Get("Mcp-Session-Id") != "session-1" {
				t.Fatalf("expected session header, got %q", r.Header.Get("Mcp-Session-Id"))
			}
			return response(http.StatusOK, header, `{"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"get-pods","description":"List pods","inputSchema":{"type":"object","properties":{"namespace":{"type":"string"}}}}]}}`), nil
		case "tools/call":
			if r.Header.Get("Mcp-Method") != "tools/call" || r.Header.Get("Mcp-Name") != "get-pods" {
				t.Fatalf("unexpected MCP headers: %q %q", r.Header.Get("Mcp-Method"), r.Header.Get("Mcp-Name"))
			}
			var params struct {
				Name      string         `json:"name"`
				Arguments map[string]any `json:"arguments"`
			}
			if err := json.Unmarshal(payload.Params, &params); err != nil {
				t.Fatal(err)
			}
			calledTool = params.Name
			calledArguments = params.Arguments
			return response(http.StatusOK, header, `{"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"ok"}],"isError":false}}`), nil
		default:
			t.Fatalf("unexpected method: %s", payload.Method)
		}
		return response(http.StatusInternalServerError, header, ""), nil
	})}

	registry := &Registry{
		clients: map[string]*Client{"kube": client},
		aliases: map[string]toolAlias{},
	}

	tools, err := registry.ListTools(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if len(tools) != 1 || tools[0].Name != "kube__get_pods" {
		t.Fatalf("unexpected tools: %#v", tools)
	}

	result, err := registry.ExecuteTool(t.Context(), "call_1", "kube__get_pods", `{"namespace":"ai"}`)
	if err != nil {
		t.Fatal(err)
	}

	if result.ToolCallID != "call_1" || result.Name != "kube__get_pods" || result.IsError {
		t.Fatalf("unexpected tool result: %#v", result)
	}
	if calledTool != "get-pods" || calledArguments["namespace"] != "ai" {
		t.Fatalf("unexpected MCP call: %q %#v", calledTool, calledArguments)
	}
	if len(methods) != 4 {
		t.Fatalf("expected initialize, notification, list, call; got %#v", methods)
	}
}

func TestRegistryFiltersTools(t *testing.T) {
	registry := NewRegistry(nil, "", time.Second, []string{"kube__get*"}, []string{"kube__get_secret*"})

	if !registry.allowed("kube__get_pods") {
		t.Fatal("expected get pods to be allowed")
	}
	if registry.allowed("kube__get_secrets") {
		t.Fatal("expected get secrets to be denied")
	}
	if registry.allowed("kube__delete_pod") {
		t.Fatal("expected delete pod to be denied")
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (fn roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) {
	return fn(r)
}

func response(statusCode int, header http.Header, body string) *http.Response {
	return &http.Response{
		StatusCode: statusCode,
		Header:     header,
		Body:       io.NopCloser(strings.NewReader(body)),
	}
}
