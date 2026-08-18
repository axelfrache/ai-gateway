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
