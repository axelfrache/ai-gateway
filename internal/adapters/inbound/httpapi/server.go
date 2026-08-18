package httpapi

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"time"

	"github.com/frachea/ai-gateway/internal/application"
	"github.com/frachea/ai-gateway/internal/domain"
)

type Server struct {
	service *application.AIService
	logger  *slog.Logger
}

func NewServer(service *application.AIService, logger *slog.Logger) *Server {
	return &Server{
		service: service,
		logger:  logger,
	}
}

func (s *Server) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", s.health)
	mux.HandleFunc("POST /v1/generate", s.generate)
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

type generateRequest struct {
	Prompt          string          `json:"prompt"`
	System          string          `json:"system,omitempty"`
	Temperature     *float64        `json:"temperature,omitempty"`
	MaxOutputTokens *int            `json:"max_output_tokens,omitempty"`
	ResponseSchema  json.RawMessage `json:"response_schema,omitempty"`
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
