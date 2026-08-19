package application

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/frachea/ai-gateway/internal/domain"
)

// Model aliases let clients pick a curated fallback chain by name — the way an
// OpenAI-compatible model picker (Open WebUI, n8n, ...) would — instead of a
// single "provider:model" candidate. They are resolved entirely inside
// AIService and never reach the outbound router.
const (
	AliasAuto  = "ai-gateway:auto"
	AliasTools = "ai-gateway:tools"
	AliasJSON  = "ai-gateway:json"
)

const aliasProvider = "ai-gateway"

var modelAliases = []struct {
	name          string
	model         string
	supportsTools bool
	supportsJSON  bool
}{
	{name: AliasAuto, model: "auto", supportsTools: false, supportsJSON: true},
	{name: AliasJSON, model: "json", supportsTools: false, supportsJSON: true},
	{name: AliasTools, model: "tools", supportsTools: true, supportsJSON: false},
}

type AIService struct {
	provider      domain.AIProvider
	models        []string
	toolModels    []string
	jsonModels    []string
	toolExecutor  domain.ToolExecutor
	maxToolRounds int
}

// extra optionally carries [toolModels, jsonModels]. Either or both may be
// omitted, in which case they default to models.
func NewAIService(provider domain.AIProvider, models []string, extra ...[]string) (*AIService, error) {
	normalized := make([]string, 0, len(models))
	seen := map[string]struct{}{}

	for _, model := range models {
		model = strings.TrimSpace(model)
		if model == "" {
			continue
		}
		if _, ok := seen[model]; ok {
			continue
		}
		seen[model] = struct{}{}
		normalized = append(normalized, model)
	}

	if provider == nil {
		return nil, errors.New("provider is required")
	}
	if len(normalized) == 0 {
		return nil, errors.New("at least one model is required")
	}

	normalizedToolModels := normalized
	if len(extra) > 0 && len(extra[0]) > 0 {
		normalizedToolModels = normalizeModels(extra[0])
	}
	if len(normalizedToolModels) == 0 {
		normalizedToolModels = normalized
	}

	normalizedJSONModels := normalized
	if len(extra) > 1 && len(extra[1]) > 0 {
		normalizedJSONModels = normalizeModels(extra[1])
	}
	if len(normalizedJSONModels) == 0 {
		normalizedJSONModels = normalized
	}

	return &AIService{
		provider:      provider,
		models:        normalized,
		toolModels:    normalizedToolModels,
		jsonModels:    normalizedJSONModels,
		maxToolRounds: 4,
	}, nil
}

// resolveAlias returns the fallback chain a model alias stands for, and
// whether model was in fact an alias.
func (s *AIService) resolveAlias(model string) ([]string, bool) {
	switch strings.ToLower(strings.TrimSpace(model)) {
	case AliasAuto:
		return append([]string(nil), s.models...), true
	case AliasTools:
		return append([]string(nil), s.toolModels...), true
	case AliasJSON:
		return append([]string(nil), s.jsonModels...), true
	default:
		return nil, false
	}
}

// expandAliases replaces any alias in models with its underlying chain.
func (s *AIService) expandAliases(models []string) []string {
	expanded := make([]string, 0, len(models))
	for _, model := range models {
		if resolved, ok := s.resolveAlias(model); ok {
			expanded = append(expanded, resolved...)
			continue
		}
		expanded = append(expanded, model)
	}
	return normalizeModels(expanded)
}

func (s *AIService) SetToolExecutor(executor domain.ToolExecutor, maxToolRounds int) {
	s.toolExecutor = executor
	if maxToolRounds > 0 {
		s.maxToolRounds = maxToolRounds
	}
}

func (s *AIService) Generate(ctx context.Context, req domain.GenerateRequest) (domain.GenerateResult, error) {
	if strings.TrimSpace(req.Prompt) == "" {
		return domain.GenerateResult{}, domain.NewError(domain.ErrorKindInvalidRequest, 0, "prompt is required", nil)
	}
	if len(req.ResponseSchema) > 0 && !validResponseSchema(req.ResponseSchema) {
		return domain.GenerateResult{}, domain.NewError(domain.ErrorKindInvalidRequest, 0, "response_schema must be a JSON object", nil)
	}

	models, err := s.modelsFor(req)
	if err != nil {
		return domain.GenerateResult{}, err
	}
	attempts := make([]domain.Attempt, 0, len(models))

	for _, model := range models {
		start := time.Now()
		response, err := s.provider.Generate(ctx, model, req)
		latencyMillis := time.Since(start).Milliseconds()

		if err == nil && strings.TrimSpace(response.Text) != "" {
			attempts = append(attempts, domain.Attempt{
				Model:         model,
				Status:        "success",
				LatencyMillis: latencyMillis,
			})
			return domain.GenerateResult{
				Model:        response.Model,
				Text:         response.Text,
				FallbackUsed: len(attempts) > 1,
				Attempts:     attempts,
			}, nil
		}

		if err == nil {
			err = domain.NewError(domain.ErrorKindTemporary, 0, "model returned an empty response", nil)
		}

		attempts = append(attempts, domain.Attempt{
			Model:         model,
			Status:        "failed",
			Error:         err.Error(),
			LatencyMillis: latencyMillis,
		})

		if !domain.Retryable(err) {
			return domain.GenerateResult{Attempts: attempts}, err
		}
	}

	return domain.GenerateResult{Attempts: attempts}, domain.NoModelAvailable(len(attempts))
}

func (s *AIService) Chat(ctx context.Context, req domain.ChatRequest) (domain.ChatResult, error) {
	if len(req.Messages) == 0 {
		return domain.ChatResult{}, domain.NewError(domain.ErrorKindInvalidRequest, 0, "messages is required", nil)
	}
	if len(req.ResponseSchema) > 0 && !validResponseSchema(req.ResponseSchema) {
		return domain.ChatResult{}, domain.NewError(domain.ErrorKindInvalidRequest, 0, "response_schema must be a JSON object", nil)
	}

	if s.toolExecutor == nil {
		return s.chatOnce(ctx, req)
	}

	current := req
	current.Messages = append([]domain.ChatMessage(nil), req.Messages...)
	allAttempts := []domain.Attempt{}
	fallbackUsed := false

	for round := 0; round <= s.maxToolRounds; round++ {
		tools, err := s.toolExecutor.ListTools(ctx)
		if err != nil {
			return domain.ChatResult{Attempts: allAttempts}, domain.NewError(domain.ErrorKindTemporary, 0, "failed to list MCP tools", err)
		}
		current.Tools, err = mergeTools(current.Tools, tools)
		if err != nil {
			return domain.ChatResult{Attempts: allAttempts}, err
		}

		result, err := s.chatOnce(ctx, current)
		allAttempts = append(allAttempts, result.Attempts...)
		if result.FallbackUsed || len(result.Attempts) > 1 {
			fallbackUsed = true
		}
		if err != nil {
			result.Attempts = allAttempts
			result.FallbackUsed = fallbackUsed
			return result, err
		}

		calls, err := toolCallsFromMessage(result.Message)
		if err != nil {
			result.Attempts = allAttempts
			result.FallbackUsed = fallbackUsed
			return result, err
		}
		executable := executableToolCalls(tools, calls)
		if len(executable) == 0 {
			result.Attempts = allAttempts
			result.FallbackUsed = fallbackUsed
			return result, nil
		}
		if len(executable) != len(calls) {
			result.Attempts = allAttempts
			result.FallbackUsed = fallbackUsed
			return result, nil
		}
		if round == s.maxToolRounds {
			result.Attempts = allAttempts
			result.FallbackUsed = fallbackUsed
			return result, domain.NewError(domain.ErrorKindTemporary, 0, "maximum tool call rounds exceeded", nil)
		}

		current.Messages = append(current.Messages, result.Message)
		for _, call := range executable {
			toolResult, err := s.toolExecutor.ExecuteTool(ctx, call.ID, call.Function.Name, call.Function.Arguments)
			if err != nil {
				return domain.ChatResult{Attempts: allAttempts}, domain.NewError(domain.ErrorKindTemporary, 0, "failed to execute MCP tool "+call.Function.Name, err)
			}
			current.Messages = append(current.Messages, domain.ChatMessage{
				Role:       "tool",
				ToolCallID: toolResult.ToolCallID,
				Name:       toolResult.Name,
				Content:    mustMarshalString(toolResult.Content),
			})
		}
	}

	return domain.ChatResult{Attempts: allAttempts, FallbackUsed: fallbackUsed}, domain.NoModelAvailable(len(allAttempts))
}

func (s *AIService) chatOnce(ctx context.Context, req domain.ChatRequest) (domain.ChatResult, error) {
	models, err := s.chatModelsFor(req)
	if err != nil {
		return domain.ChatResult{}, err
	}
	attempts := make([]domain.Attempt, 0, len(models))

	chatProvider, ok := s.provider.(domain.ChatProvider)
	if !ok {
		return domain.ChatResult{}, domain.NewError(domain.ErrorKindInvalidRequest, 0, "chat completions are not supported by configured provider", nil)
	}

	for _, model := range models {
		start := time.Now()
		response, err := chatProvider.Chat(ctx, model, req)
		latencyMillis := time.Since(start).Milliseconds()

		if err == nil && chatResponseHasValue(response) {
			attempts = append(attempts, domain.Attempt{
				Model:         model,
				Status:        "success",
				LatencyMillis: latencyMillis,
			})
			return domain.ChatResult{
				Model:        response.Model,
				Message:      response.Message,
				FinishReason: response.FinishReason,
				FallbackUsed: len(attempts) > 1,
				Attempts:     attempts,
			}, nil
		}

		if err == nil {
			err = domain.NewError(domain.ErrorKindTemporary, 0, "model returned an empty response", nil)
		}

		attempts = append(attempts, domain.Attempt{
			Model:         model,
			Status:        "failed",
			Error:         err.Error(),
			LatencyMillis: latencyMillis,
		})

		if !domain.Retryable(err) {
			return domain.ChatResult{Attempts: attempts}, err
		}
	}

	return domain.ChatResult{Attempts: attempts}, domain.NoModelAvailable(len(attempts))
}

func (s *AIService) Models() []domain.ModelInfo {
	candidates := append([]string(nil), s.models...)
	candidates = append(candidates, s.toolModels...)
	candidates = append(candidates, s.jsonModels...)
	candidates = normalizeModels(candidates)
	toolModels := map[string]struct{}{}
	for _, model := range s.toolModels {
		toolModels[model] = struct{}{}
	}
	jsonModels := map[string]struct{}{}
	for _, model := range s.jsonModels {
		jsonModels[model] = struct{}{}
	}

	models := make([]domain.ModelInfo, 0, len(candidates)+len(modelAliases))
	order := 0
	for _, alias := range modelAliases {
		order++
		models = append(models, domain.ModelInfo{
			Name:          alias.name,
			Provider:      aliasProvider,
			Model:         alias.model,
			Order:         order,
			SupportsTools: alias.supportsTools,
			SupportsJSON:  alias.supportsJSON,
		})
	}
	for _, model := range candidates {
		order++
		provider, name := splitModel(model)
		_, supportsTools := toolModels[model]
		_, supportsJSON := jsonModels[model]
		models = append(models, domain.ModelInfo{
			Name:          model,
			Provider:      provider,
			Model:         name,
			Order:         order,
			SupportsTools: supportsTools,
			SupportsJSON:  supportsJSON,
		})
	}
	return models
}

func (s *AIService) CheckModels(ctx context.Context, models []string) []domain.ModelCheck {
	candidates := normalizeModels(models)
	if len(candidates) == 0 {
		candidates = append([]string(nil), s.models...)
	}
	candidates = s.expandAliases(candidates)

	maxTokens := 128
	request := domain.GenerateRequest{
		Prompt:          "Return an object with answer set to ok.",
		System:          "You must answer using the provided JSON schema.",
		MaxOutputTokens: &maxTokens,
		ResponseSchema:  json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}`),
	}

	results := make([]domain.ModelCheck, 0, len(candidates))
	for _, candidate := range candidates {
		start := time.Now()
		_, err := s.provider.Generate(ctx, candidate, request)
		provider, model := splitModel(candidate)
		check := domain.ModelCheck{
			Name:          candidate,
			Provider:      provider,
			Model:         model,
			Status:        "available",
			LatencyMillis: time.Since(start).Milliseconds(),
		}
		if err != nil {
			check.Status = "unavailable"
			check.Error = err.Error()
		}
		results = append(results, check)
	}
	return results
}

func (s *AIService) modelsFor(req domain.GenerateRequest) ([]string, error) {
	if strings.TrimSpace(req.Model) != "" && len(req.Models) > 0 {
		return nil, domain.NewError(domain.ErrorKindInvalidRequest, 0, "use either model or models, not both", nil)
	}
	if strings.TrimSpace(req.Model) != "" {
		if models, ok := s.resolveAlias(req.Model); ok {
			return models, nil
		}
		return []string{strings.TrimSpace(req.Model)}, nil
	}
	if models := normalizeModels(req.Models); len(models) > 0 {
		return models, nil
	}
	return append([]string(nil), s.models...), nil
}

func (s *AIService) chatModelsFor(req domain.ChatRequest) ([]string, error) {
	if strings.TrimSpace(req.Model) != "" && len(req.Models) > 0 {
		return nil, domain.NewError(domain.ErrorKindInvalidRequest, 0, "use either model or models, not both", nil)
	}
	if strings.TrimSpace(req.Model) != "" {
		if models, ok := s.resolveAlias(req.Model); ok {
			return models, nil
		}
		return []string{strings.TrimSpace(req.Model)}, nil
	}
	if models := normalizeModels(req.Models); len(models) > 0 {
		return models, nil
	}
	if chatRequestHasTools(req) {
		return append([]string(nil), s.toolModels...), nil
	}
	return append([]string(nil), s.models...), nil
}

func validResponseSchema(schema json.RawMessage) bool {
	if !json.Valid(schema) {
		return false
	}

	var object map[string]any
	if err := json.Unmarshal(schema, &object); err != nil {
		return false
	}
	return len(object) > 0
}

func normalizeModels(models []string) []string {
	normalized := make([]string, 0, len(models))
	seen := map[string]struct{}{}
	for _, model := range models {
		model = strings.TrimSpace(model)
		if model == "" {
			continue
		}
		if _, ok := seen[model]; ok {
			continue
		}
		seen[model] = struct{}{}
		normalized = append(normalized, model)
	}
	return normalized
}

func chatRequestHasTools(req domain.ChatRequest) bool {
	return rawJSONHasValue(req.Tools) && !rawJSONEquals(req.Tools, "[]") && !rawJSONEquals(req.Tools, "{}")
}

func chatResponseHasValue(response domain.ChatResponse) bool {
	if rawJSONHasValue(response.Message.ToolCalls) {
		return true
	}
	if !rawJSONHasValue(response.Message.Content) {
		return false
	}
	var text string
	if err := json.Unmarshal(response.Message.Content, &text); err == nil {
		return strings.TrimSpace(text) != ""
	}
	return true
}

func rawJSONHasValue(raw json.RawMessage) bool {
	trimmed := strings.TrimSpace(string(raw))
	return trimmed != "" && trimmed != "null"
}

func rawJSONEquals(raw json.RawMessage, value string) bool {
	return strings.TrimSpace(string(raw)) == value
}

type chatToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

func mergeTools(raw json.RawMessage, tools []domain.ToolDefinition) (json.RawMessage, error) {
	var values []json.RawMessage
	if rawJSONHasValue(raw) {
		if err := json.Unmarshal(raw, &values); err != nil {
			return nil, domain.NewError(domain.ErrorKindInvalidRequest, 0, "tools must be a JSON array", err)
		}
	}

	seen := map[string]struct{}{}
	for _, rawTool := range values {
		name, err := openAIToolName(rawTool)
		if err != nil {
			return nil, err
		}
		if name != "" {
			seen[name] = struct{}{}
		}
	}

	for _, tool := range tools {
		name := strings.TrimSpace(tool.Name)
		if name == "" {
			continue
		}
		if _, ok := seen[name]; ok {
			continue
		}
		parameters := tool.Parameters
		if !rawJSONHasValue(parameters) {
			parameters = json.RawMessage(`{"type":"object","properties":{}}`)
		}
		rawTool, err := json.Marshal(map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        name,
				"description": tool.Description,
				"parameters":  json.RawMessage(parameters),
			},
		})
		if err != nil {
			return nil, domain.NewError(domain.ErrorKindInvalidRequest, 0, "failed to encode MCP tool", err)
		}
		values = append(values, rawTool)
		seen[name] = struct{}{}
	}

	if len(values) == 0 {
		return nil, nil
	}
	merged, err := json.Marshal(values)
	if err != nil {
		return nil, domain.NewError(domain.ErrorKindInvalidRequest, 0, "failed to encode tools", err)
	}
	return merged, nil
}

func openAIToolName(raw json.RawMessage) (string, error) {
	var tool struct {
		Type     string `json:"type"`
		Function struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	if err := json.Unmarshal(raw, &tool); err != nil {
		return "", domain.NewError(domain.ErrorKindInvalidRequest, 0, "tools must contain JSON objects", err)
	}
	if tool.Type != "function" {
		return "", nil
	}
	return strings.TrimSpace(tool.Function.Name), nil
}

func toolCallsFromMessage(message domain.ChatMessage) ([]chatToolCall, error) {
	if !rawJSONHasValue(message.ToolCalls) {
		return nil, nil
	}
	var calls []chatToolCall
	if err := json.Unmarshal(message.ToolCalls, &calls); err != nil {
		return nil, domain.NewError(domain.ErrorKindTemporary, 0, "model returned invalid tool calls", err)
	}
	for i := range calls {
		if strings.TrimSpace(calls[i].ID) == "" {
			calls[i].ID = fmt.Sprintf("call_%d", i)
		}
	}
	return calls, nil
}

func executableToolCalls(tools []domain.ToolDefinition, calls []chatToolCall) []chatToolCall {
	available := map[string]struct{}{}
	for _, tool := range tools {
		available[tool.Name] = struct{}{}
	}
	result := make([]chatToolCall, 0, len(calls))
	for _, call := range calls {
		if _, ok := available[call.Function.Name]; ok {
			result = append(result, call)
		}
	}
	return result
}

func mustMarshalString(value string) json.RawMessage {
	encoded, err := json.Marshal(value)
	if err != nil {
		return json.RawMessage(`""`)
	}
	return encoded
}

func splitModel(candidate string) (string, string) {
	if provider, model, ok := strings.Cut(candidate, ":"); ok {
		return strings.TrimSpace(provider), strings.TrimSpace(model)
	}
	return "", strings.TrimSpace(candidate)
}
