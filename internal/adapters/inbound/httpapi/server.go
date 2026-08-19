package httpapi

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/frachea/ai-gateway/internal/application"
	"github.com/frachea/ai-gateway/internal/domain"
)

type Server struct {
	service *application.AIService
	logger  *slog.Logger
	apiKeys []string
}

func NewServer(service *application.AIService, logger *slog.Logger, apiKeys []string) *Server {
	return &Server{
		service: service,
		logger:  logger,
		apiKeys: compactStrings(apiKeys),
	}
}

func (s *Server) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", s.health)
	mux.Handle("GET /api/tags", s.requireAuth(http.HandlerFunc(s.ollamaTags)))
	mux.Handle("POST /api/generate", s.requireAuth(http.HandlerFunc(s.ollamaGenerate)))
	mux.Handle("GET /v1/models", s.requireAuth(http.HandlerFunc(s.models)))
	mux.Handle("POST /v1/models/check", s.requireAuth(http.HandlerFunc(s.checkModels)))
	mux.Handle("POST /v1/generate", s.requireAuth(http.HandlerFunc(s.generate)))
	mux.Handle("POST /v1/chat/completions", s.requireAuth(http.HandlerFunc(s.chatCompletions)))
	return requestLogger(s.logger, mux)
}

func (s *Server) health(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *Server) generate(w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()

	var payload generateRequest
	if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body", nil)
		return
	}

	result, err := s.service.Generate(r.Context(), domain.GenerateRequest{
		Prompt:          payload.Prompt,
		System:          payload.System,
		Model:           payload.Model,
		Models:          payload.Models,
		Temperature:     payload.Temperature,
		MaxOutputTokens: payload.MaxOutputTokens,
		ResponseSchema:  payload.ResponseSchema,
	})
	if err != nil {
		statusCode := statusCodeFor(err)
		writeError(w, statusCode, err.Error(), result.Attempts)
		return
	}

	writeJSON(w, http.StatusOK, result)
}

func (s *Server) models(w http.ResponseWriter, _ *http.Request) {
	models := s.service.Models()
	data := make([]openAIModel, 0, len(models))
	for _, model := range models {
		owner := model.Provider
		if owner == "" {
			owner = "ai-gateway"
		}
		data = append(data, openAIModel{
			ID:      model.Name,
			Object:  "model",
			Created: 0,
			OwnedBy: owner,
		})
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"object": "list",
		"data":   data,
		"models": models,
	})
}

func (s *Server) checkModels(w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()

	var payload checkModelsRequest
	if r.Body != nil {
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil && !errors.Is(err, io.EOF) {
			writeError(w, http.StatusBadRequest, "invalid JSON body", nil)
			return
		}
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"models": s.service.CheckModels(r.Context(), payload.Models),
	})
}

func (s *Server) ollamaTags(w http.ResponseWriter, _ *http.Request) {
	models := s.service.Models()
	items := make([]ollamaModel, 0, len(models))
	modifiedAt := time.Now().UTC()

	for _, model := range models {
		family := model.Provider
		if family == "" {
			family = "unknown"
		}

		items = append(items, ollamaModel{
			Name:       model.Name,
			Model:      model.Name,
			ModifiedAt: modifiedAt,
			Details: ollamaModelDetails{
				Format: "remote",
				Family: family,
				Families: []string{
					family,
				},
				ParameterSize:     "",
				QuantizationLevel: "",
			},
		})
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"models": items,
	})
}

func (s *Server) ollamaGenerate(w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()

	var payload ollamaGenerateRequest
	if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body", nil)
		return
	}

	if payload.Stream != nil && *payload.Stream {
		writeError(w, http.StatusBadRequest, "streaming is not supported", nil)
		return
	}

	responseSchema, err := ollamaFormatToSchema(payload.Format)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error(), nil)
		return
	}

	start := time.Now()
	result, err := s.service.Generate(r.Context(), domain.GenerateRequest{
		Prompt:          payload.Prompt,
		System:          payload.System,
		Model:           payload.Model,
		Models:          payload.Models,
		Temperature:     payload.Options.Temperature,
		MaxOutputTokens: payload.Options.NumPredict,
		ResponseSchema:  responseSchema,
	})
	if err != nil {
		statusCode := statusCodeFor(err)
		writeError(w, statusCode, err.Error(), result.Attempts)
		return
	}

	writeJSON(w, http.StatusOK, ollamaGenerateResponse{
		Model:         result.Model,
		CreatedAt:     time.Now().UTC(),
		Response:      result.Text,
		Done:          true,
		DoneReason:    "stop",
		TotalDuration: time.Since(start).Nanoseconds(),
	})
}

func (s *Server) chatCompletions(w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()

	var payload chatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body", nil)
		return
	}

	responseSchema, err := openAIResponseFormatToSchema(payload.ResponseFormat)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error(), nil)
		return
	}

	maxOutputTokens := payload.MaxTokens
	if maxOutputTokens == nil {
		maxOutputTokens = payload.MaxCompletionTokens
	}

	result, err := s.service.Chat(r.Context(), domain.ChatRequest{
		Messages:        domainMessages(payload.Messages),
		Model:           payload.Model,
		Models:          payload.Models,
		Temperature:     payload.Temperature,
		MaxOutputTokens: maxOutputTokens,
		ResponseSchema:  responseSchema,
		Tools:           payload.Tools,
		ToolChoice:      payload.ToolChoice,
	})
	if err != nil {
		statusCode := statusCodeFor(err)
		writeError(w, statusCode, err.Error(), result.Attempts)
		return
	}

	if payload.Stream {
		writeChatCompletionStream(w, result)
		return
	}

	writeJSON(w, http.StatusOK, newChatCompletionResponse(result))
}

type generateRequest struct {
	Prompt          string          `json:"prompt"`
	System          string          `json:"system,omitempty"`
	Model           string          `json:"model,omitempty"`
	Models          []string        `json:"models,omitempty"`
	Temperature     *float64        `json:"temperature,omitempty"`
	MaxOutputTokens *int            `json:"max_output_tokens,omitempty"`
	ResponseSchema  json.RawMessage `json:"response_schema,omitempty"`
}

type checkModelsRequest struct {
	Models []string `json:"models,omitempty"`
}

type openAIModel struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

type chatCompletionRequest struct {
	Model               string          `json:"model,omitempty"`
	Models              []string        `json:"models,omitempty"`
	Messages            []chatMessage   `json:"messages"`
	Temperature         *float64        `json:"temperature,omitempty"`
	MaxTokens           *int            `json:"max_tokens,omitempty"`
	MaxCompletionTokens *int            `json:"max_completion_tokens,omitempty"`
	Stream              bool            `json:"stream,omitempty"`
	ResponseFormat      json.RawMessage `json:"response_format,omitempty"`
	Tools               json.RawMessage `json:"tools,omitempty"`
	ToolChoice          json.RawMessage `json:"tool_choice,omitempty"`
}

type chatMessage struct {
	Role       string          `json:"role"`
	Content    json.RawMessage `json:"content"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
	Name       string          `json:"name,omitempty"`
	ToolCalls  json.RawMessage `json:"tool_calls,omitempty"`
}

type chatCompletionResponse struct {
	ID      string                 `json:"id"`
	Object  string                 `json:"object"`
	Created int64                  `json:"created"`
	Model   string                 `json:"model"`
	Choices []chatCompletionChoice `json:"choices"`
	Usage   chatCompletionUsage    `json:"usage"`
}

type chatCompletionChoice struct {
	Index        int                    `json:"index"`
	Message      *chatCompletionMessage `json:"message,omitempty"`
	Delta        *chatCompletionDelta   `json:"delta,omitempty"`
	FinishReason *string                `json:"finish_reason"`
}

type chatCompletionMessage struct {
	Role      string          `json:"role"`
	Content   any             `json:"content"`
	ToolCalls json.RawMessage `json:"tool_calls,omitempty"`
}

type chatCompletionDelta struct {
	Role      string          `json:"role,omitempty"`
	Content   any             `json:"content,omitempty"`
	ToolCalls json.RawMessage `json:"tool_calls,omitempty"`
}

type chatCompletionUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type ollamaGenerateRequest struct {
	Model   string          `json:"model,omitempty"`
	Models  []string        `json:"models,omitempty"`
	Prompt  string          `json:"prompt"`
	System  string          `json:"system,omitempty"`
	Stream  *bool           `json:"stream,omitempty"`
	Format  json.RawMessage `json:"format,omitempty"`
	Options ollamaOptions   `json:"options,omitempty"`
}

type ollamaOptions struct {
	Temperature *float64 `json:"temperature,omitempty"`
	NumPredict  *int     `json:"num_predict,omitempty"`
}

type ollamaGenerateResponse struct {
	Model         string    `json:"model"`
	CreatedAt     time.Time `json:"created_at"`
	Response      string    `json:"response"`
	Done          bool      `json:"done"`
	DoneReason    string    `json:"done_reason,omitempty"`
	TotalDuration int64     `json:"total_duration"`
}

type ollamaModel struct {
	Name       string             `json:"name"`
	Model      string             `json:"model"`
	ModifiedAt time.Time          `json:"modified_at"`
	Size       int64              `json:"size"`
	Digest     string             `json:"digest"`
	Details    ollamaModelDetails `json:"details"`
}

type ollamaModelDetails struct {
	Format            string   `json:"format"`
	Family            string   `json:"family"`
	Families          []string `json:"families"`
	ParameterSize     string   `json:"parameter_size"`
	QuantizationLevel string   `json:"quantization_level"`
}

func ollamaFormatToSchema(format json.RawMessage) (json.RawMessage, error) {
	if len(format) == 0 {
		return nil, nil
	}

	var value any
	if err := json.Unmarshal(format, &value); err != nil {
		return nil, errors.New("format must be json or a JSON schema object")
	}

	switch typed := value.(type) {
	case string:
		if typed != "json" {
			return nil, errors.New("format must be json or a JSON schema object")
		}
		return json.RawMessage(`{"type":"object"}`), nil
	case map[string]any:
		if len(typed) == 0 {
			return nil, errors.New("format must be json or a JSON schema object")
		}
		return format, nil
	default:
		return nil, errors.New("format must be json or a JSON schema object")
	}
}

func openAIResponseFormatToSchema(format json.RawMessage) (json.RawMessage, error) {
	if len(format) == 0 {
		return nil, nil
	}

	var value map[string]json.RawMessage
	if err := json.Unmarshal(format, &value); err != nil {
		return nil, errors.New("response_format must be a JSON object")
	}

	var formatType string
	if rawJSONHasValue(value["type"]) {
		if err := json.Unmarshal(value["type"], &formatType); err != nil {
			return nil, errors.New("response_format.type must be a string")
		}
	}

	switch formatType {
	case "", "text":
		return nil, nil
	case "json_object":
		return json.RawMessage(`{"type":"object"}`), nil
	case "json_schema":
		var jsonSchema struct {
			Schema json.RawMessage `json:"schema"`
		}
		if err := json.Unmarshal(value["json_schema"], &jsonSchema); err != nil || len(jsonSchema.Schema) == 0 {
			return nil, errors.New("response_format.json_schema.schema is required")
		}
		return jsonSchema.Schema, nil
	default:
		return nil, errors.New("unsupported response_format.type")
	}
}

func domainMessages(messages []chatMessage) []domain.ChatMessage {
	result := make([]domain.ChatMessage, 0, len(messages))
	for _, message := range messages {
		result = append(result, domain.ChatMessage{
			Role:       message.Role,
			Content:    message.Content,
			ToolCallID: message.ToolCallID,
			Name:       message.Name,
			ToolCalls:  message.ToolCalls,
		})
	}
	return result
}

func chatMessageContentValue(raw json.RawMessage) any {
	if !rawJSONHasValue(raw) {
		return nil
	}
	var value any
	if err := json.Unmarshal(raw, &value); err != nil {
		return nil
	}
	return value
}

func newChatCompletionResponse(result domain.ChatResult) chatCompletionResponse {
	reason := strings.TrimSpace(result.FinishReason)
	if reason == "" {
		reason = "stop"
	}
	content := chatMessageContentValue(result.Message.Content)
	return chatCompletionResponse{
		ID:      chatCompletionID(),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   result.Model,
		Choices: []chatCompletionChoice{
			{
				Index: 0,
				Message: &chatCompletionMessage{
					Role:      "assistant",
					Content:   content,
					ToolCalls: result.Message.ToolCalls,
				},
				FinishReason: &reason,
			},
		},
		Usage: chatCompletionUsage{},
	}
}

func writeChatCompletionStream(w http.ResponseWriter, result domain.ChatResult) {
	id := chatCompletionID()
	created := time.Now().Unix()
	reason := strings.TrimSpace(result.FinishReason)
	if reason == "" {
		reason = "stop"
	}
	content := chatMessageContentValue(result.Message.Content)

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	writeSSE(w, map[string]any{
		"id":      id,
		"object":  "chat.completion.chunk",
		"created": created,
		"model":   result.Model,
		"choices": []chatCompletionChoice{
			{
				Index: 0,
				Delta: &chatCompletionDelta{
					Role:      "assistant",
					Content:   content,
					ToolCalls: result.Message.ToolCalls,
				},
				FinishReason: nil,
			},
		},
	})
	writeSSE(w, map[string]any{
		"id":      id,
		"object":  "chat.completion.chunk",
		"created": created,
		"model":   result.Model,
		"choices": []chatCompletionChoice{
			{
				Index:        0,
				Delta:        &chatCompletionDelta{},
				FinishReason: &reason,
			},
		},
	})
	_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	if flusher, ok := w.(http.Flusher); ok {
		flusher.Flush()
	}
}

func writeSSE(w http.ResponseWriter, value any) {
	body, err := json.Marshal(value)
	if err != nil {
		return
	}
	_, _ = fmt.Fprintf(w, "data: %s\n\n", body)
	if flusher, ok := w.(http.Flusher); ok {
		flusher.Flush()
	}
}

func chatCompletionID() string {
	return fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
}

func rawJSONHasValue(raw json.RawMessage) bool {
	trimmed := strings.TrimSpace(string(raw))
	return trimmed != "" && trimmed != "null"
}

func statusCodeFor(err error) int {
	var aiErr *domain.AIError
	if !errors.As(err, &aiErr) {
		return http.StatusInternalServerError
	}

	switch aiErr.Kind {
	case domain.ErrorKindInvalidRequest:
		return http.StatusBadRequest
	case domain.ErrorKindAuth:
		return http.StatusBadGateway
	case domain.ErrorKindRateLimited, domain.ErrorKindTemporary, domain.ErrorKindModelUnavailable, domain.ErrorKindNoModel:
		return http.StatusServiceUnavailable
	case domain.ErrorKindSafety:
		return http.StatusUnprocessableEntity
	default:
		return http.StatusInternalServerError
	}
}

func writeJSON(w http.ResponseWriter, statusCode int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	_ = json.NewEncoder(w).Encode(value)
}

func writeError(w http.ResponseWriter, statusCode int, message string, attempts []domain.Attempt) {
	writeJSON(w, statusCode, map[string]any{
		"error":    message,
		"attempts": attempts,
	})
}

func (s *Server) requireAuth(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if s.authorized(r) {
			next.ServeHTTP(w, r)
			return
		}

		w.Header().Set("WWW-Authenticate", "Bearer")
		writeError(w, http.StatusUnauthorized, "unauthorized", nil)
	})
}

func (s *Server) authorized(r *http.Request) bool {
	token := bearerToken(r.Header.Get("Authorization"))
	if token == "" {
		token = strings.TrimSpace(r.Header.Get("X-API-Key"))
	}
	if token == "" {
		return false
	}

	for _, apiKey := range s.apiKeys {
		if subtle.ConstantTimeCompare([]byte(token), []byte(apiKey)) == 1 {
			return true
		}
	}
	return false
}

func bearerToken(value string) string {
	const prefix = "Bearer "
	if !strings.HasPrefix(value, prefix) {
		return ""
	}
	return strings.TrimSpace(strings.TrimPrefix(value, prefix))
}

func compactStrings(values []string) []string {
	compacted := make([]string, 0, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value != "" {
			compacted = append(compacted, value)
		}
	}
	return compacted
}

func requestLogger(logger *slog.Logger, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		logger.InfoContext(context.Background(), "http request",
			"method", r.Method,
			"path", r.URL.Path,
			"duration_ms", time.Since(start).Milliseconds(),
		)
	})
}
