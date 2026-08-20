package gemini

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/frachea/ai-gateway/internal/domain"
)

type Client struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
}

func NewClient(apiKey, baseURL string, timeout time.Duration) *Client {
	return &Client{
		apiKey:  apiKey,
		baseURL: strings.TrimRight(baseURL, "/"),
		httpClient: &http.Client{
			Timeout: timeout,
		},
	}
}

func (c *Client) Generate(ctx context.Context, model string, req domain.GenerateRequest) (domain.GenerateResponse, error) {
	payload := generateContentRequest{
		Contents: []content{
			{
				Role: "user",
				Parts: []part{
					{Text: req.Prompt},
				},
			},
		},
	}

	if strings.TrimSpace(req.System) != "" {
		payload.SystemInstruction = &content{
			Parts: []part{{Text: req.System}},
		}
	}

	if req.Temperature != nil || req.MaxOutputTokens != nil || len(req.ResponseSchema) > 0 {
		payload.GenerationConfig = &generationConfig{
			Temperature:     req.Temperature,
			MaxOutputTokens: req.MaxOutputTokens,
		}
		if len(req.ResponseSchema) > 0 {
			responseSchema, err := normalizeSchema(req.ResponseSchema)
			if err != nil {
				return domain.GenerateResponse{}, domain.NewError(domain.ErrorKindInvalidRequest, 0, "failed to normalize Gemini response schema", err)
			}
			payload.GenerationConfig.ResponseMimeType = "application/json"
			payload.GenerationConfig.ResponseSchema = responseSchema
		}
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return domain.GenerateResponse{}, domain.NewError(domain.ErrorKindInvalidRequest, 0, "failed to encode Gemini request", err)
	}

	endpoint := fmt.Sprintf("%s/models/%s:generateContent?key=%s", c.baseURL, url.PathEscape(model), url.QueryEscape(c.apiKey))
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return domain.GenerateResponse{}, domain.NewError(domain.ErrorKindInvalidRequest, 0, "failed to build Gemini request", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	httpResp, err := c.httpClient.Do(httpReq)
	if err != nil {
		if errors.Is(err, context.Canceled) {
			return domain.GenerateResponse{}, domain.NewError(domain.ErrorKindTemporary, 0, "request was canceled", err)
		}
		if errors.Is(err, context.DeadlineExceeded) {
			return domain.GenerateResponse{}, domain.NewError(domain.ErrorKindTemporary, 0, "request timed out", err)
		}
		return domain.GenerateResponse{}, domain.NewError(domain.ErrorKindTemporary, 0, "Gemini request failed", err)
	}
	defer httpResp.Body.Close()

	responseBody, err := io.ReadAll(io.LimitReader(httpResp.Body, 4<<20))
	if err != nil {
		return domain.GenerateResponse{}, domain.NewError(domain.ErrorKindTemporary, httpResp.StatusCode, "failed to read Gemini response", err)
	}

	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		return domain.GenerateResponse{}, classifyGeminiError(httpResp.StatusCode, responseBody)
	}

	var geminiResp generateContentResponse
	if err := json.Unmarshal(responseBody, &geminiResp); err != nil {
		return domain.GenerateResponse{}, domain.NewError(domain.ErrorKindTemporary, httpResp.StatusCode, "failed to decode Gemini response", err)
	}

	text := geminiResp.Text()
	if text == "" {
		if geminiResp.PromptFeedback.BlockReason != "" {
			return domain.GenerateResponse{}, domain.NewError(domain.ErrorKindSafety, httpResp.StatusCode, "Gemini blocked the prompt: "+geminiResp.PromptFeedback.BlockReason, nil)
		}
		if finishReason := geminiResp.FirstFinishReason(); finishReason == "SAFETY" || finishReason == "RECITATION" {
			return domain.GenerateResponse{}, domain.NewError(domain.ErrorKindSafety, httpResp.StatusCode, "Gemini stopped generation: "+finishReason, nil)
		}
		return domain.GenerateResponse{}, domain.NewError(domain.ErrorKindTemporary, httpResp.StatusCode, "Gemini returned no text", nil)
	}

	return domain.GenerateResponse{
		Model: model,
		Text:  text,
	}, nil
}

func (c *Client) Chat(ctx context.Context, model string, req domain.ChatRequest) (domain.ChatResponse, error) {
	contents, systemInstruction, err := contentsFromMessages(req.Messages)
	if err != nil {
		return domain.ChatResponse{}, domain.NewError(domain.ErrorKindInvalidRequest, 0, "failed to convert Gemini messages", err)
	}

	payload := generateContentRequest{
		Contents: contents,
	}

	if strings.TrimSpace(systemInstruction) != "" {
		payload.SystemInstruction = &content{
			Parts: []part{{Text: systemInstruction}},
		}
	}

	if req.Temperature != nil || req.MaxOutputTokens != nil || len(req.ResponseSchema) > 0 {
		payload.GenerationConfig = &generationConfig{
			Temperature:     req.Temperature,
			MaxOutputTokens: req.MaxOutputTokens,
		}
		if len(req.ResponseSchema) > 0 {
			responseSchema, err := normalizeSchema(req.ResponseSchema)
			if err != nil {
				return domain.ChatResponse{}, domain.NewError(domain.ErrorKindInvalidRequest, 0, "failed to normalize Gemini response schema", err)
			}
			payload.GenerationConfig.ResponseMimeType = "application/json"
			payload.GenerationConfig.ResponseSchema = responseSchema
		}
	}

	if rawJSONHasValue(req.Tools) {
		declarations, err := functionDeclarationsFromTools(req.Tools)
		if err != nil {
			return domain.ChatResponse{}, domain.NewError(domain.ErrorKindInvalidRequest, 0, "failed to convert Gemini tools", err)
		}
		if len(declarations) > 0 {
			payload.Tools = []tool{{FunctionDeclarations: declarations}}
		}
	}

	if rawJSONHasValue(req.ToolChoice) {
		toolConfig, err := toolConfigFromChoice(req.ToolChoice)
		if err != nil {
			return domain.ChatResponse{}, domain.NewError(domain.ErrorKindInvalidRequest, 0, "failed to convert Gemini tool choice", err)
		}
		payload.ToolConfig = toolConfig
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return domain.ChatResponse{}, domain.NewError(domain.ErrorKindInvalidRequest, 0, "failed to encode Gemini request", err)
	}

	endpoint := fmt.Sprintf("%s/models/%s:generateContent?key=%s", c.baseURL, url.PathEscape(model), url.QueryEscape(c.apiKey))
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return domain.ChatResponse{}, domain.NewError(domain.ErrorKindInvalidRequest, 0, "failed to build Gemini request", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	httpResp, err := c.httpClient.Do(httpReq)
	if err != nil {
		if errors.Is(err, context.Canceled) {
			return domain.ChatResponse{}, domain.NewError(domain.ErrorKindTemporary, 0, "request was canceled", err)
		}
		if errors.Is(err, context.DeadlineExceeded) {
			return domain.ChatResponse{}, domain.NewError(domain.ErrorKindTemporary, 0, "request timed out", err)
		}
		return domain.ChatResponse{}, domain.NewError(domain.ErrorKindTemporary, 0, "Gemini request failed", err)
	}
	defer httpResp.Body.Close()

	responseBody, err := io.ReadAll(io.LimitReader(httpResp.Body, 4<<20))
	if err != nil {
		return domain.ChatResponse{}, domain.NewError(domain.ErrorKindTemporary, httpResp.StatusCode, "failed to read Gemini response", err)
	}

	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		return domain.ChatResponse{}, classifyGeminiError(httpResp.StatusCode, responseBody)
	}

	var geminiResp generateContentResponse
	if err := json.Unmarshal(responseBody, &geminiResp); err != nil {
		return domain.ChatResponse{}, domain.NewError(domain.ErrorKindTemporary, httpResp.StatusCode, "failed to decode Gemini response", err)
	}

	message, finishReason, err := geminiResp.ChatMessage()
	if err != nil {
		if geminiResp.PromptFeedback.BlockReason != "" {
			return domain.ChatResponse{}, domain.NewError(domain.ErrorKindSafety, httpResp.StatusCode, "Gemini blocked the prompt: "+geminiResp.PromptFeedback.BlockReason, nil)
		}
		if finishReason := geminiResp.FirstFinishReason(); finishReason == "SAFETY" || finishReason == "RECITATION" {
			return domain.ChatResponse{}, domain.NewError(domain.ErrorKindSafety, httpResp.StatusCode, "Gemini stopped generation: "+finishReason, nil)
		}
		return domain.ChatResponse{}, domain.NewError(domain.ErrorKindTemporary, httpResp.StatusCode, err.Error(), nil)
	}

	return domain.ChatResponse{
		Model:        model,
		Message:      message,
		FinishReason: finishReason,
	}, nil
}

type generateContentRequest struct {
	Contents          []content         `json:"contents"`
	SystemInstruction *content          `json:"systemInstruction,omitempty"`
	GenerationConfig  *generationConfig `json:"generationConfig,omitempty"`
	Tools             []tool            `json:"tools,omitempty"`
	ToolConfig        *toolConfig       `json:"toolConfig,omitempty"`
}

type content struct {
	Role  string `json:"role,omitempty"`
	Parts []part `json:"parts"`
}

type part struct {
	Text             string            `json:"text,omitempty"`
	FunctionCall     *functionCall     `json:"functionCall,omitempty"`
	FunctionResponse *functionResponse `json:"functionResponse,omitempty"`
	ThoughtSignature string            `json:"thoughtSignature,omitempty"`
}

type generationConfig struct {
	Temperature      *float64        `json:"temperature,omitempty"`
	MaxOutputTokens  *int            `json:"maxOutputTokens,omitempty"`
	ResponseMimeType string          `json:"responseMimeType,omitempty"`
	ResponseSchema   json.RawMessage `json:"responseSchema,omitempty"`
}

type tool struct {
	FunctionDeclarations []functionDeclaration `json:"functionDeclarations,omitempty"`
}

type functionDeclaration struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

type toolConfig struct {
	FunctionCallingConfig *functionCallingConfig `json:"functionCallingConfig,omitempty"`
}

type functionCallingConfig struct {
	Mode                 string   `json:"mode,omitempty"`
	AllowedFunctionNames []string `json:"allowedFunctionNames,omitempty"`
}

type functionCall struct {
	Name string          `json:"name"`
	Args json.RawMessage `json:"args,omitempty"`
}

type functionResponse struct {
	Name     string          `json:"name"`
	Response json.RawMessage `json:"response"`
}

type generateContentResponse struct {
	Candidates []struct {
		Content struct {
			Parts []struct {
				Text             string        `json:"text"`
				FunctionCall     *functionCall `json:"functionCall"`
				ThoughtSignature string        `json:"thoughtSignature"`
			} `json:"parts"`
		} `json:"content"`
		FinishReason string `json:"finishReason"`
	} `json:"candidates"`
	PromptFeedback struct {
		BlockReason string `json:"blockReason"`
	} `json:"promptFeedback"`
}

func (r generateContentResponse) Text() string {
	var builder strings.Builder
	for _, candidate := range r.Candidates {
		for _, part := range candidate.Content.Parts {
			if part.Text == "" {
				continue
			}
			if builder.Len() > 0 {
				builder.WriteString("\n")
			}
			builder.WriteString(part.Text)
		}
		if builder.Len() > 0 {
			break
		}
	}
	return strings.TrimSpace(builder.String())
}

func (r generateContentResponse) FirstFinishReason() string {
	for _, candidate := range r.Candidates {
		if candidate.FinishReason != "" {
			return candidate.FinishReason
		}
	}
	return ""
}

func (r generateContentResponse) ChatMessage() (domain.ChatMessage, string, error) {
	for _, candidate := range r.Candidates {
		var textParts []string
		var toolCalls []openAIToolCall

		for _, part := range candidate.Content.Parts {
			if strings.TrimSpace(part.Text) != "" {
				textParts = append(textParts, part.Text)
			}
			if part.FunctionCall != nil && strings.TrimSpace(part.FunctionCall.Name) != "" {
				arguments := "{}"
				if rawJSONHasValue(part.FunctionCall.Args) {
					arguments = string(part.FunctionCall.Args)
				}
				toolCalls = append(toolCalls, openAIToolCall{
					ID:   fmt.Sprintf("call_%d_%d", time.Now().UnixNano(), len(toolCalls)),
					Type: "function",
					Function: openAIFunctionCall{
						Name:      part.FunctionCall.Name,
						Arguments: arguments,
					},
					ThoughtSignature: part.ThoughtSignature,
				})
			}
		}

		text := strings.TrimSpace(strings.Join(textParts, "\n"))
		content, err := json.Marshal(text)
		if err != nil {
			return domain.ChatMessage{}, "", err
		}

		message := domain.ChatMessage{
			Role:    "assistant",
			Content: content,
		}

		finishReason := finishReasonFromGemini(candidate.FinishReason)
		if len(toolCalls) > 0 {
			rawToolCalls, err := json.Marshal(toolCalls)
			if err != nil {
				return domain.ChatMessage{}, "", err
			}
			message.Content = json.RawMessage("null")
			message.ToolCalls = rawToolCalls
			finishReason = "tool_calls"
		}

		if text != "" || len(toolCalls) > 0 {
			return message, finishReason, nil
		}
	}

	return domain.ChatMessage{}, "", errors.New("Gemini returned no message")
}

type openAITool struct {
	Type     string `json:"type"`
	Function struct {
		Name        string          `json:"name"`
		Description string          `json:"description"`
		Parameters  json.RawMessage `json:"parameters"`
	} `json:"function"`
}

type openAIToolCall struct {
	ID               string             `json:"id,omitempty"`
	Type             string             `json:"type"`
	Function         openAIFunctionCall `json:"function"`
	ThoughtSignature string             `json:"thought_signature,omitempty"`
}

type openAIFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

func contentsFromMessages(messages []domain.ChatMessage) ([]content, string, error) {
	contents := make([]content, 0, len(messages))
	systemParts := []string{}
	toolCallNames := map[string]string{}

	for _, message := range messages {
		role := strings.TrimSpace(strings.ToLower(message.Role))
		text, err := messageContentText(message.Content)
		if err != nil {
			return nil, "", err
		}

		switch role {
		case "system", "developer":
			if strings.TrimSpace(text) != "" {
				systemParts = append(systemParts, text)
			}
		case "user":
			if strings.TrimSpace(text) != "" {
				contents = append(contents, content{Role: "user", Parts: []part{{Text: text}}})
			}
		case "assistant":
			item := content{Role: "model"}
			if strings.TrimSpace(text) != "" {
				item.Parts = append(item.Parts, part{Text: text})
			}
			calls, err := toolCallsFromRaw(message.ToolCalls)
			if err != nil {
				return nil, "", err
			}
			for _, call := range calls {
				args, err := functionArgsFromOpenAI(call.Function.Arguments)
				if err != nil {
					return nil, "", err
				}
				signature := strings.TrimSpace(call.ThoughtSignature)
				if signature == "" {
					// Gemini requires every replayed functionCall part to carry a
					// thoughtSignature. Calls that didn't originate from Gemini (a
					// different fallback provider, a client-injected tool call, or a
					// parallel call Gemini itself left unsigned) get this documented
					// placeholder instead of a real one.
					signature = "skip_thought_signature_validator"
				}
				item.Parts = append(item.Parts, part{
					FunctionCall:     &functionCall{Name: call.Function.Name, Args: args},
					ThoughtSignature: signature,
				})
				if strings.TrimSpace(call.ID) != "" {
					toolCallNames[call.ID] = call.Function.Name
				}
			}
			if len(item.Parts) > 0 {
				contents = append(contents, item)
			}
		case "tool":
			name := strings.TrimSpace(message.Name)
			if name == "" {
				name = toolCallNames[strings.TrimSpace(message.ToolCallID)]
			}
			if name == "" {
				return nil, "", errors.New("tool message name is required")
			}
			response, err := functionResponseContent(text)
			if err != nil {
				return nil, "", err
			}
			contents = append(contents, content{
				Role: "user",
				Parts: []part{
					{FunctionResponse: &functionResponse{Name: name, Response: response}},
				},
			})
		default:
			return nil, "", fmt.Errorf("unsupported message role %q", role)
		}
	}

	if len(contents) == 0 {
		return nil, "", errors.New("messages must include user, assistant, or tool content")
	}

	return contents, strings.TrimSpace(strings.Join(systemParts, "\n\n")), nil
}

func functionDeclarationsFromTools(raw json.RawMessage) ([]functionDeclaration, error) {
	var tools []openAITool
	if err := json.Unmarshal(raw, &tools); err != nil {
		return nil, err
	}

	declarations := make([]functionDeclaration, 0, len(tools))
	for _, item := range tools {
		if strings.TrimSpace(item.Type) != "function" {
			continue
		}
		name := strings.TrimSpace(item.Function.Name)
		if name == "" {
			return nil, errors.New("tool function name is required")
		}
		parameters := item.Function.Parameters
		if !rawJSONHasValue(parameters) {
			parameters = json.RawMessage(`{"type":"object","properties":{}}`)
		}
		normalized, err := normalizeSchema(parameters)
		if err != nil {
			return nil, err
		}
		declarations = append(declarations, functionDeclaration{
			Name:        name,
			Description: item.Function.Description,
			Parameters:  normalized,
		})
	}
	return declarations, nil
}

func toolConfigFromChoice(raw json.RawMessage) (*toolConfig, error) {
	if !rawJSONHasValue(raw) {
		return nil, nil
	}

	var choice string
	if err := json.Unmarshal(raw, &choice); err == nil {
		switch choice {
		case "auto":
			return &toolConfig{FunctionCallingConfig: &functionCallingConfig{Mode: "AUTO"}}, nil
		case "none":
			return &toolConfig{FunctionCallingConfig: &functionCallingConfig{Mode: "NONE"}}, nil
		case "required", "any":
			return &toolConfig{FunctionCallingConfig: &functionCallingConfig{Mode: "ANY"}}, nil
		default:
			return nil, fmt.Errorf("unsupported tool_choice %q", choice)
		}
	}

	var object struct {
		Type     string `json:"type"`
		Function struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	if err := json.Unmarshal(raw, &object); err != nil {
		return nil, err
	}
	if object.Type == "function" && strings.TrimSpace(object.Function.Name) != "" {
		return &toolConfig{
			FunctionCallingConfig: &functionCallingConfig{
				Mode:                 "ANY",
				AllowedFunctionNames: []string{strings.TrimSpace(object.Function.Name)},
			},
		}, nil
	}
	return nil, errors.New("unsupported tool_choice")
}

func toolCallsFromRaw(raw json.RawMessage) ([]openAIToolCall, error) {
	if !rawJSONHasValue(raw) {
		return nil, nil
	}
	var calls []openAIToolCall
	if err := json.Unmarshal(raw, &calls); err != nil {
		return nil, err
	}
	return calls, nil
}

func functionArgsFromOpenAI(arguments string) (json.RawMessage, error) {
	arguments = strings.TrimSpace(arguments)
	if arguments == "" {
		return json.RawMessage("{}"), nil
	}
	if json.Valid([]byte(arguments)) {
		var value any
		if err := json.Unmarshal([]byte(arguments), &value); err != nil {
			return nil, err
		}
		encoded, err := json.Marshal(value)
		if err != nil {
			return nil, err
		}
		return encoded, nil
	}
	encoded, err := json.Marshal(map[string]string{"value": arguments})
	if err != nil {
		return nil, err
	}
	return encoded, nil
}

func functionResponseContent(text string) (json.RawMessage, error) {
	var value any = text
	if json.Valid([]byte(text)) {
		_ = json.Unmarshal([]byte(text), &value)
	}
	encoded, err := json.Marshal(map[string]any{"result": value})
	if err != nil {
		return nil, err
	}
	return encoded, nil
}

func messageContentText(raw json.RawMessage) (string, error) {
	if !rawJSONHasValue(raw) {
		return "", nil
	}
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return text, nil
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &parts); err != nil {
		return "", errors.New("message content must be a string or text parts")
	}
	values := make([]string, 0, len(parts))
	for _, part := range parts {
		if part.Type == "text" && strings.TrimSpace(part.Text) != "" {
			values = append(values, part.Text)
		}
	}
	return strings.TrimSpace(strings.Join(values, "\n")), nil
}

func finishReasonFromGemini(reason string) string {
	switch strings.ToUpper(strings.TrimSpace(reason)) {
	case "MAX_TOKENS":
		return "length"
	case "SAFETY", "RECITATION":
		return "content_filter"
	default:
		return "stop"
	}
}

func rawJSONHasValue(raw json.RawMessage) bool {
	trimmed := strings.TrimSpace(string(raw))
	return trimmed != "" && trimmed != "null"
}

var geminiSchemaKeywords = map[string]bool{
	"type":        true,
	"format":      true,
	"description": true,
	"nullable":    true,
	"enum":        true,
	"maxItems":    true,
	"minItems":    true,
	"properties":  true,
	"required":    true,
	"items":       true,
	"anyOf":       true,
}

func normalizeSchema(schema json.RawMessage) (json.RawMessage, error) {
	var value any
	if err := json.Unmarshal(schema, &value); err != nil {
		return nil, err
	}
	sanitizeSchemaForGemini(value)

	normalized, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	return normalized, nil
}

func sanitizeSchemaForGemini(value any) {
	typed, ok := value.(map[string]any)
	if !ok {
		return
	}

	if constValue, ok := typed["const"]; ok {
		delete(typed, "const")
		if _, hasEnum := typed["enum"]; !hasEnum {
			typed["enum"] = []any{constValue}
		}
	}
	for key := range typed {
		if !geminiSchemaKeywords[key] {
			delete(typed, key)
		}
	}

	if properties, ok := typed["properties"].(map[string]any); ok {
		for _, propSchema := range properties {
			sanitizeSchemaForGemini(propSchema)
		}
	}
	if items, ok := typed["items"]; ok {
		sanitizeSchemaForGemini(items)
	}
	if anyOf, ok := typed["anyOf"].([]any); ok {
		for _, sub := range anyOf {
			sanitizeSchemaForGemini(sub)
		}
	}
}

type errorResponse struct {
	Error struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
		Status  string `json:"status"`
	} `json:"error"`
}

func classifyGeminiError(statusCode int, body []byte) error {
	var parsed errorResponse
	_ = json.Unmarshal(body, &parsed)

	message := strings.TrimSpace(parsed.Error.Message)
	if message == "" {
		message = http.StatusText(statusCode)
	}
	if message == "" {
		message = fmt.Sprintf("Gemini returned HTTP %d", statusCode)
	}

	status := strings.ToUpper(parsed.Error.Status)

	switch {
	case statusCode == http.StatusTooManyRequests || status == "RESOURCE_EXHAUSTED":
		return domain.NewError(domain.ErrorKindRateLimited, statusCode, message, nil)
	case statusCode == http.StatusUnauthorized || statusCode == http.StatusForbidden:
		return domain.NewError(domain.ErrorKindAuth, statusCode, message, nil)
	case (statusCode == http.StatusBadRequest || status == "INVALID_ARGUMENT") && tokenLimitError(message):
		return domain.NewError(domain.ErrorKindModelUnavailable, statusCode, message, nil)
	case statusCode == http.StatusBadRequest || status == "INVALID_ARGUMENT":
		return domain.NewError(domain.ErrorKindInvalidRequest, statusCode, message, nil)
	case statusCode == http.StatusNotFound || status == "NOT_FOUND":
		return domain.NewError(domain.ErrorKindModelUnavailable, statusCode, message, nil)
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
