package mcpsmoke

import (
	"encoding/json"
	"net/http"
	"time"
)

type Server struct {
	startedAt time.Time
}

type requestEnvelope struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type responseEnvelope struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Result  any             `json:"result,omitempty"`
	Error   *rpcError       `json:"error,omitempty"`
}

type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

func NewServer() *Server {
	return &Server{startedAt: time.Now().UTC()}
}

func (s *Server) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", s.health)
	mux.HandleFunc("/mcp", s.mcp)
	return mux
}

func (s *Server) health(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"status": "ok"})
}

func (s *Server) mcp(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	var req requestEnvelope
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeRPC(w, responseEnvelope{
			JSONRPC: "2.0",
			ID:      req.ID,
			Error:   &rpcError{Code: -32700, Message: "parse error"},
		})
		return
	}

	switch req.Method {
	case "initialize":
		writeRPC(w, responseEnvelope{
			JSONRPC: "2.0",
			ID:      req.ID,
			Result: map[string]any{
				"protocolVersion": "2025-11-25",
				"capabilities": map[string]any{
					"tools": map[string]any{
						"listChanged": false,
					},
				},
				"serverInfo": map[string]any{
					"name":    "mcp-smoke",
					"version": "0.1.0",
				},
			},
		})
	case "notifications/initialized":
		w.WriteHeader(http.StatusAccepted)
	case "tools/list":
		writeRPC(w, responseEnvelope{
			JSONRPC: "2.0",
			ID:      req.ID,
			Result: map[string]any{
				"tools": []map[string]any{
					{
						"name":        "status",
						"description": "Return the smoke MCP server status.",
						"inputSchema": map[string]any{
							"type":       "object",
							"properties": map[string]any{},
						},
					},
					{
						"name":        "echo",
						"description": "Return the provided message and payload.",
						"inputSchema": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"message": map[string]any{"type": "string"},
								"payload": map[string]any{"type": "object"},
							},
						},
					},
				},
			},
		})
	case "tools/call":
		s.callTool(w, req)
	default:
		writeRPC(w, responseEnvelope{
			JSONRPC: "2.0",
			ID:      req.ID,
			Error:   &rpcError{Code: -32601, Message: "method not found"},
		})
	}
}

func (s *Server) callTool(w http.ResponseWriter, req requestEnvelope) {
	var params struct {
		Name      string         `json:"name"`
		Arguments map[string]any `json:"arguments"`
	}
	if len(req.Params) > 0 {
		_ = json.Unmarshal(req.Params, &params)
	}

	switch params.Name {
	case "status":
		result := map[string]any{
			"ok":         true,
			"name":       "mcp-smoke",
			"started_at": s.startedAt.Format(time.RFC3339),
		}
		writeToolResult(w, req.ID, result, false)
	case "echo":
		result := map[string]any{
			"message": params.Arguments["message"],
			"payload": params.Arguments["payload"],
		}
		writeToolResult(w, req.ID, result, false)
	default:
		writeToolResult(w, req.ID, map[string]any{"error": "unknown tool"}, true)
	}
}

func writeToolResult(w http.ResponseWriter, id json.RawMessage, result map[string]any, isError bool) {
	text, _ := json.Marshal(result)
	writeRPC(w, responseEnvelope{
		JSONRPC: "2.0",
		ID:      id,
		Result: map[string]any{
			"content": []map[string]any{
				{
					"type": "text",
					"text": string(text),
				},
			},
			"structuredContent": result,
			"isError":           isError,
		},
	})
}

func writeRPC(w http.ResponseWriter, response responseEnvelope) {
	writeJSON(w, http.StatusOK, response)
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}
