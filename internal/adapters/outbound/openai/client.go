package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/frachea/ai-gateway/internal/domain"
)

type Client struct {
	name           string
	apiKey         string
	baseURL        string
	maxTokensField string
	openRouter     bool
	httpClient     *http.Client
}

func NewClient(name, apiKey, baseURL, maxTokensField string, timeout time.Duration) *Client {
	if maxTokensField == "" {
		maxTokensField = "max_tokens"
	}
	return &Client{
		name:           name,
		apiKey:         apiKey,
		baseURL:        strings.TrimRight(baseURL, "/"),
		maxTokensField: maxTokensField,
		httpClient: &http.Client{
			Timeout: timeout,
		},
	}
}

func NewOpenRouterClient(apiKey, baseURL string, timeout time.Duration) *Client {
	client := NewClient("OpenRouter", apiKey, baseURL, "max_tokens", timeout)
	client.openRouter = true
	return client
}

func (c *Client) Generate(ctx context.Context, model string, req domain.GenerateRequest) (domain.GenerateResponse, error) {
	messageContent, err := json.Marshal(req.Prompt)
	if err != nil {
		return domain.GenerateResponse{}, domain.NewError(domain.ErrorKindInvalidRequest, 0, "failed to encode "+c.name+" prompt", err)
	}

	payload := map[string]any{
		"model": model,
		"messages": []message{
			{Role: "user", Content: messageContent},
		},
	}

	if strings.TrimSpace(req.System) != "" {
		systemContent, err := json.Marshal(req.System)
		if err != nil {
			return domain.GenerateResponse{}, domain.NewError(domain.ErrorKindInvalidRequest, 0, "failed to encode "+c.name+" system prompt", err)
		}
		payload["messages"] = []message{
			{Role: "system", Content: systemContent},
			{Role: "user", Content: messageContent},
		}
	}
	if req.Temperature != nil {
		payload["temperature"] = *req.Temperature
	}
	if req.MaxOutputTokens != nil {
		payload[c.maxTokensField] = *req.MaxOutputTokens
	}
	if len(req.ResponseSchema) > 0 {
		payload["response_format"] = map[string]any{
			"type": "json_schema",
			"json_schema": map[string]any{
				"name":   "response",
				"strict": true,
				"schema": json.RawMessage(req.ResponseSchema),
			},
		}
		if c.openRouter {
			payload["provider"] = map[string]any{
				"require_parameters": true,
			}
		}
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return domain.GenerateResponse{}, domain.NewError(domain.ErrorKindInvalidRequest, 0, "failed to encode "+c.name+" request", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return domain.GenerateResponse{}, domain.NewError(domain.ErrorKindInvalidRequest, 0, "failed to build "+c.name+" request", err)
	}
	httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	httpReq.Header.Set("Content-Type", "application/json")

	httpResp, err := c.httpClient.Do(httpReq)
	if err != nil {
		if errors.Is(err, context.Canceled) {
			return domain.GenerateResponse{}, domain.NewError(domain.ErrorKindTemporary, 0, "request was canceled", err)
		}
		if errors.Is(err, context.DeadlineExceeded) {
			return domain.GenerateResponse{}, domain.NewError(domain.ErrorKindTemporary, 0, "request timed out", err)
		}
		return domain.GenerateResponse{}, domain.NewError(domain.ErrorKindTemporary, 0, c.name+" request failed", err)
	}
	defer httpResp.Body.Close()

	responseBody, err := io.ReadAll(io.LimitReader(httpResp.Body, 4<<20))
	if err != nil {
		return domain.GenerateResponse{}, domain.NewError(domain.ErrorKindTemporary, httpResp.StatusCode, "failed to read "+c.name+" response", err)
	}

	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		return domain.GenerateResponse{}, classifyError(c.name, httpResp.StatusCode, responseBody)
	}

	var parsed chatCompletionResponse
	if err := json.Unmarshal(responseBody, &parsed); err != nil {
		return domain.GenerateResponse{}, domain.NewError(domain.ErrorKindTemporary, httpResp.StatusCode, "failed to decode "+c.name+" response", err)
	}

	text := parsed.Text()
	if text == "" {
		return domain.GenerateResponse{}, domain.NewError(domain.ErrorKindTemporary, httpResp.StatusCode, c.name+" returned no text", nil)
	}

	return domain.GenerateResponse{Model: model, Text: text}, nil
}

func (c *Client) Chat(ctx context.Context, model string, req domain.ChatRequest) (domain.ChatResponse, error) {
	payload := map[string]any{
		"model":    model,
		"messages": openAIMessages(req.Messages),
	}

	if req.Temperature != nil {
		payload["temperature"] = *req.Temperature
	}
	if req.MaxOutputTokens != nil {
		payload[c.maxTokensField] = *req.MaxOutputTokens
	}
	if len(req.ResponseSchema) > 0 {
		payload["response_format"] = map[string]any{
			"type": "json_schema",
			"json_schema": map[string]any{
				"name":   "response",
				"strict": true,
				"schema": json.RawMessage(req.ResponseSchema),
			},
		}
		if c.openRouter {
			payload["provider"] = map[string]any{
				"require_parameters": true,
			}
		}
	}
	if rawJSONHasValue(req.Tools) {
		payload["tools"] = json.RawMessage(req.Tools)
	}
	if rawJSONHasValue(req.ToolChoice) {
		payload["tool_choice"] = c.toolChoice(req.ToolChoice)
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return domain.ChatResponse{}, domain.NewError(domain.ErrorKindInvalidRequest, 0, "failed to encode "+c.name+" request", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return domain.ChatResponse{}, domain.NewError(domain.ErrorKindInvalidRequest, 0, "failed to build "+c.name+" request", err)
	}
	httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	httpReq.Header.Set("Content-Type", "application/json")

	httpResp, err := c.httpClient.Do(httpReq)
	if err != nil {
		if errors.Is(err, context.Canceled) {
			return domain.ChatResponse{}, domain.NewError(domain.ErrorKindTemporary, 0, "request was canceled", err)
		}
		if errors.Is(err, context.DeadlineExceeded) {
			return domain.ChatResponse{}, domain.NewError(domain.ErrorKindTemporary, 0, "request timed out", err)
		}
		return domain.ChatResponse{}, domain.NewError(domain.ErrorKindTemporary, 0, c.name+" request failed", err)
	}
	defer httpResp.Body.Close()

	responseBody, err := io.ReadAll(io.LimitReader(httpResp.Body, 4<<20))
	if err != nil {
		return domain.ChatResponse{}, domain.NewError(domain.ErrorKindTemporary, httpResp.StatusCode, "failed to read "+c.name+" response", err)
	}

	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		return domain.ChatResponse{}, classifyError(c.name, httpResp.StatusCode, responseBody)
	}

	var parsed chatCompletionResponse
	if err := json.Unmarshal(responseBody, &parsed); err != nil {
		return domain.ChatResponse{}, domain.NewError(domain.ErrorKindTemporary, httpResp.StatusCode, "failed to decode "+c.name+" response", err)
	}

	message, finishReason := parsed.Message()
	if !chatMessageHasValue(message) {
		return domain.ChatResponse{}, domain.NewError(domain.ErrorKindTemporary, httpResp.StatusCode, c.name+" returned no message", nil)
	}

	return domain.ChatResponse{
		Model:        model,
		Message:      message,
		FinishReason: finishReason,
	}, nil
}

type message struct {
	Role       string          `json:"role"`
	Content    json.RawMessage `json:"content,omitempty"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
	Name       string          `json:"name,omitempty"`
	ToolCalls  json.RawMessage `json:"tool_calls,omitempty"`
}

type chatCompletionResponse struct {
	Choices []struct {
		Message      message `json:"message"`
		FinishReason string  `json:"finish_reason"`
	} `json:"choices"`
}

func (r chatCompletionResponse) Text() string {
	for _, choice := range r.Choices {
		var text string
		if err := json.Unmarshal(choice.Message.Content, &text); err == nil && strings.TrimSpace(text) != "" {
			return strings.TrimSpace(text)
		}
	}
	return ""
}

func (r chatCompletionResponse) Message() (domain.ChatMessage, string) {
	for _, choice := range r.Choices {
		content := choice.Message.Content
		if len(content) == 0 {
			content = json.RawMessage("null")
		}
		message := domain.ChatMessage{
			Role:      strings.TrimSpace(choice.Message.Role),
			Content:   content,
			ToolCalls: choice.Message.ToolCalls,
		}
		if message.Role == "" {
			message.Role = "assistant"
		}
		if chatMessageHasValue(message) {
			return message, strings.TrimSpace(choice.FinishReason)
		}
	}
	return domain.ChatMessage{}, ""
}

func openAIMessages(messages []domain.ChatMessage) []message {
	result := make([]message, 0, len(messages))
	for _, item := range messages {
		content := item.Content
		if len(content) == 0 {
			content = json.RawMessage("null")
		}
		result = append(result, message{
			Role:       item.Role,
			Content:    content,
			ToolCallID: item.ToolCallID,
			Name:       item.Name,
			ToolCalls:  item.ToolCalls,
		})
	}
	return result
}

func chatMessageHasValue(message domain.ChatMessage) bool {
	if rawJSONHasValue(message.ToolCalls) {
		return true
	}
	if !rawJSONHasValue(message.Content) {
		return false
	}
	var text string
	if err := json.Unmarshal(message.Content, &text); err == nil {
		return strings.TrimSpace(text) != ""
	}
	return true
}

func rawJSONHasValue(raw json.RawMessage) bool {
	trimmed := strings.TrimSpace(string(raw))
	return trimmed != "" && trimmed != "null"
}

func (c *Client) toolChoice(raw json.RawMessage) any {
	if c.name == "Mistral" && rawJSONEqualsString(raw, "required") {
		return "any"
	}
	return json.RawMessage(raw)
}

func rawJSONEqualsString(raw json.RawMessage, expected string) bool {
	var value string
	if err := json.Unmarshal(raw, &value); err != nil {
		return false
	}
	return value == expected
}

type errorResponse struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    any    `json:"code"`
	} `json:"error"`
	Message string `json:"message"`
}

func classifyError(provider string, statusCode int, body []byte) error {
	var parsed errorResponse
	_ = json.Unmarshal(body, &parsed)

	message := strings.TrimSpace(parsed.Error.Message)
	if message == "" {
		message = strings.TrimSpace(parsed.Message)
	}
	if message == "" {
		message = http.StatusText(statusCode)
	}
	if message == "" {
		message = fmt.Sprintf("%s returned HTTP %d", provider, statusCode)
	}

	switch {
	case statusCode == http.StatusTooManyRequests:
		return domain.NewError(domain.ErrorKindRateLimited, statusCode, message, nil)
	case statusCode == http.StatusUnauthorized || statusCode == http.StatusForbidden:
		return domain.NewError(domain.ErrorKindAuth, statusCode, message, nil)
	case statusCode == http.StatusNotFound:
		return domain.NewError(domain.ErrorKindModelUnavailable, statusCode, message, nil)
	case statusCode == http.StatusBadRequest && tokenLimitError(message):
		return domain.NewError(domain.ErrorKindModelUnavailable, statusCode, message, nil)
	case statusCode == http.StatusBadRequest:
		return domain.NewError(domain.ErrorKindInvalidRequest, statusCode, message, nil)
	case statusCode >= 500:
		return domain.NewError(domain.ErrorKindTemporary, statusCode, message, nil)
	default:
		return domain.NewError(domain.ErrorKindUnknown, statusCode, message, nil)
	}
}

func tokenLimitError(message string) bool {
	normalized := strings.ToLower(message)
	patterns := []string{
		"context length",
		"context limit",
		"context window",
		"maximum context",
		"max context",
		"too many tokens",
		"token limit",
		"tokens exceed",
		"exceeds the maximum",
	}
	for _, pattern := range patterns {
		if strings.Contains(normalized, pattern) {
			return true
		}
	}
	return false
}
