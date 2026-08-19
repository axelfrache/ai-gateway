package mcp

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/frachea/ai-gateway/internal/domain"
)

type ServerConfig struct {
	Name        string
	URL         string
	BearerToken string
}

type Registry struct {
	clients map[string]*Client
	aliases map[string]toolAlias
	allow   []string
	deny    []string
	mu      sync.RWMutex
}

type Client struct {
	name            string
	endpoint        string
	bearerToken     string
	protocolVersion string
	httpClient      *http.Client
	nextID          atomic.Int64
	mu              sync.Mutex
	initialized     bool
	sessionID       string
}

type Tool struct {
	Name        string          `json:"name"`
	Title       string          `json:"title,omitempty"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"inputSchema,omitempty"`
}

type CallResult struct {
	Content           []map[string]any `json:"content,omitempty"`
	StructuredContent json.RawMessage  `json:"structuredContent,omitempty"`
	IsError           bool             `json:"isError,omitempty"`
	Extra             map[string]any   `json:"-"`
}

type toolAlias struct {
	server string
	name   string
}

func NewRegistry(configs []ServerConfig, protocolVersion string, timeout time.Duration, allow, deny []string) *Registry {
	if protocolVersion == "" {
		protocolVersion = "2025-11-25"
	}
	clients := map[string]*Client{}
	for _, cfg := range configs {
		name := sanitizeName(cfg.Name)
		if name == "" || strings.TrimSpace(cfg.URL) == "" {
			continue
		}
		clients[name] = NewClient(name, cfg.URL, cfg.BearerToken, protocolVersion, timeout)
	}
	return &Registry{
		clients: clients,
		aliases: map[string]toolAlias{},
		allow:   compact(allow),
		deny:    compact(deny),
	}
}

func NewClient(name, endpoint, bearerToken, protocolVersion string, timeout time.Duration) *Client {
	if timeout <= 0 {
		timeout = 20 * time.Second
	}
	return &Client{
		name:            name,
		endpoint:        strings.TrimSpace(endpoint),
		bearerToken:     strings.TrimSpace(bearerToken),
		protocolVersion: protocolVersion,
		httpClient:      &http.Client{Timeout: timeout},
	}
}

func (r *Registry) ListTools(ctx context.Context) ([]domain.ToolDefinition, error) {
	definitions := []domain.ToolDefinition{}
	aliases := map[string]toolAlias{}

	for serverName, client := range r.clients {
		tools, err := client.ListTools(ctx)
		if err != nil {
			return nil, err
		}
		for _, tool := range tools {
			exposedName := sanitizeName(serverName + "__" + tool.Name)
			if exposedName == "" || !r.allowed(exposedName) {
				continue
			}
			description := strings.TrimSpace(tool.Description)
			if description == "" {
				description = strings.TrimSpace(tool.Title)
			}
			parameters := tool.InputSchema
			if !rawJSONHasValue(parameters) {
				parameters = json.RawMessage(`{"type":"object","properties":{}}`)
			}
			definitions = append(definitions, domain.ToolDefinition{
				Name:        exposedName,
				Description: description,
				Parameters:  parameters,
			})
			aliases[exposedName] = toolAlias{server: serverName, name: tool.Name}
		}
	}

	r.mu.Lock()
	r.aliases = aliases
	r.mu.Unlock()

	return definitions, nil
}

func (r *Registry) ExecuteTool(ctx context.Context, toolCallID, name, arguments string) (domain.ToolResult, error) {
	r.mu.RLock()
	alias, ok := r.aliases[name]
	r.mu.RUnlock()
	if !ok {
		return domain.ToolResult{}, fmt.Errorf("unknown MCP tool %q", name)
	}
	if !r.allowed(name) {
		return domain.ToolResult{}, fmt.Errorf("MCP tool %q is not allowed", name)
	}
	client := r.clients[alias.server]
	if client == nil {
		return domain.ToolResult{}, fmt.Errorf("MCP server %q is not configured", alias.server)
	}
	var args map[string]any
	if strings.TrimSpace(arguments) != "" {
		if err := json.Unmarshal([]byte(arguments), &args); err != nil {
			return domain.ToolResult{}, err
		}
	}
	result, err := client.CallTool(ctx, alias.name, args)
	if err != nil {
		return domain.ToolResult{}, err
	}
	content, err := json.Marshal(result)
	if err != nil {
		return domain.ToolResult{}, err
	}
	return domain.ToolResult{
		ToolCallID: toolCallID,
		Name:       name,
		Content:    string(content),
		IsError:    result.IsError,
	}, nil
}

func (r *Registry) allowed(name string) bool {
	for _, pattern := range r.deny {
		if matchPattern(pattern, name) {
			return false
		}
	}
	if len(r.allow) == 0 {
		return true
	}
	for _, pattern := range r.allow {
		if matchPattern(pattern, name) {
			return true
		}
	}
	return false
}

func (c *Client) ListTools(ctx context.Context) ([]Tool, error) {
	if err := c.ensureInitialized(ctx); err != nil {
		return nil, err
	}

	tools := []Tool{}
	cursor := ""
	for {
		params := map[string]any{}
		if cursor != "" {
			params["cursor"] = cursor
		}
		var result struct {
			Tools      []Tool `json:"tools"`
			NextCursor string `json:"nextCursor"`
		}
		if err := c.request(ctx, "tools/list", "", params, &result); err != nil {
			return nil, err
		}
		tools = append(tools, result.Tools...)
		if strings.TrimSpace(result.NextCursor) == "" {
			break
		}
		cursor = result.NextCursor
	}
	return tools, nil
}

func (c *Client) CallTool(ctx context.Context, name string, args map[string]any) (CallResult, error) {
	if err := c.ensureInitialized(ctx); err != nil {
		return CallResult{}, err
	}
	params := map[string]any{
		"name": name,
	}
	if args != nil {
		params["arguments"] = args
	}
	var result CallResult
	if err := c.request(ctx, "tools/call", name, params, &result); err != nil {
		return CallResult{}, err
	}
	return result, nil
}

func (c *Client) ensureInitialized(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.initialized {
		return nil
	}
	params := map[string]any{
		"protocolVersion": c.protocolVersion,
		"capabilities":    map[string]any{},
		"clientInfo": map[string]string{
			"name":    "ai-gateway",
			"version": "0.1.0",
		},
	}
	var result map[string]any
	if err := c.requestLocked(ctx, "initialize", "", params, &result); err != nil {
		return err
	}
	_ = c.notificationLocked(ctx, "notifications/initialized", nil)
	c.initialized = true
	return nil
}

func (c *Client) request(ctx context.Context, method, name string, params any, result any) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.requestLocked(ctx, method, name, params, result)
}

func (c *Client) requestLocked(ctx context.Context, method, name string, params any, result any) error {
	id := c.nextID.Add(1)
	payload := map[string]any{
		"jsonrpc": "2.0",
		"id":      id,
		"method":  method,
	}
	if params != nil {
		payload["params"] = params
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	responseBody, err := c.post(ctx, method, name, body)
	if err != nil {
		return err
	}
	return decodeResponse(responseBody, result)
}

func (c *Client) notificationLocked(ctx context.Context, method string, params any) error {
	payload := map[string]any{
		"jsonrpc": "2.0",
		"method":  method,
	}
	if params != nil {
		payload["params"] = params
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	_, err = c.post(ctx, method, "", body)
	return err
}

func (c *Client) post(ctx context.Context, method, name string, body []byte) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")
	req.Header.Set("Mcp-Method", method)
	if name != "" {
		req.Header.Set("Mcp-Name", name)
	}
	if c.sessionID != "" {
		req.Header.Set("Mcp-Session-Id", c.sessionID)
	}
	if c.bearerToken != "" {
		req.Header.Set("Authorization", "Bearer "+c.bearerToken)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if sessionID := strings.TrimSpace(resp.Header.Get("Mcp-Session-Id")); sessionID != "" {
		c.sessionID = sessionID
	}

	if resp.StatusCode == http.StatusAccepted {
		return []byte(`{"jsonrpc":"2.0","result":{}}`), nil
	}

	responseBody, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("MCP server %s returned HTTP %d: %s", c.name, resp.StatusCode, strings.TrimSpace(string(responseBody)))
	}

	contentType, _, _ := mime.ParseMediaType(resp.Header.Get("Content-Type"))
	if contentType == "text/event-stream" {
		return firstSSEData(responseBody)
	}
	return responseBody, nil
}

func decodeResponse(body []byte, result any) error {
	var envelope struct {
		Result json.RawMessage `json:"result"`
		Error  *struct {
			Code    int             `json:"code"`
			Message string          `json:"message"`
			Data    json.RawMessage `json:"data,omitempty"`
		} `json:"error,omitempty"`
	}
	if err := json.Unmarshal(body, &envelope); err != nil {
		return err
	}
	if envelope.Error != nil {
		return errors.New(envelope.Error.Message)
	}
	if result == nil || !rawJSONHasValue(envelope.Result) {
		return nil
	}
	return json.Unmarshal(envelope.Result, result)
}

func firstSSEData(body []byte) ([]byte, error) {
	scanner := bufio.NewScanner(bytes.NewReader(body))
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		value := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if value != "" {
			return []byte(value), nil
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return nil, errors.New("MCP SSE response did not contain data")
}

func sanitizeName(value string) string {
	value = strings.TrimSpace(value)
	var builder strings.Builder
	for _, r := range value {
		switch {
		case r >= 'a' && r <= 'z':
			builder.WriteRune(r)
		case r >= 'A' && r <= 'Z':
			builder.WriteRune(r)
		case r >= '0' && r <= '9':
			builder.WriteRune(r)
		case r == '_' || r == '-' || r == '.':
			builder.WriteRune('_')
		default:
			builder.WriteRune('_')
		}
	}
	return strings.Trim(builder.String(), "_")
}

func matchPattern(pattern, value string) bool {
	pattern = strings.TrimSpace(pattern)
	value = strings.TrimSpace(value)
	if pattern == "" {
		return false
	}
	if pattern == "*" || pattern == value {
		return true
	}
	if strings.HasSuffix(pattern, "*") && strings.HasPrefix(value, strings.TrimSuffix(pattern, "*")) {
		return true
	}
	if strings.HasPrefix(pattern, "*") && strings.HasSuffix(value, strings.TrimPrefix(pattern, "*")) {
		return true
	}
	return false
}

func compact(values []string) []string {
	result := make([]string, 0, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value != "" {
			result = append(result, value)
		}
	}
	return result
}

func rawJSONHasValue(raw json.RawMessage) bool {
	trimmed := strings.TrimSpace(string(raw))
	return trimmed != "" && trimmed != "null"
}
