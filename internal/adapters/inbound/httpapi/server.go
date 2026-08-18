package httpapi

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
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
	writeJSON(w, http.StatusOK, map[string]any{
		"models": s.service.Models(),
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
