package application

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"time"

	"github.com/frachea/ai-gateway/internal/domain"
)

type AIService struct {
	provider domain.AIProvider
	models   []string
}

func NewAIService(provider domain.AIProvider, models []string) (*AIService, error) {
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

	return &AIService{
		provider: provider,
		models:   normalized,
	}, nil
}

func (s *AIService) Generate(ctx context.Context, req domain.GenerateRequest) (domain.GenerateResult, error) {
	if strings.TrimSpace(req.Prompt) == "" {
		return domain.GenerateResult{}, domain.NewError(domain.ErrorKindInvalidRequest, 0, "prompt is required", nil)
	}
	if len(req.ResponseSchema) > 0 && !validResponseSchema(req.ResponseSchema) {
		return domain.GenerateResult{}, domain.NewError(domain.ErrorKindInvalidRequest, 0, "response_schema must be a JSON object", nil)
	}

	attempts := make([]domain.Attempt, 0, len(s.models))

	for _, model := range s.models {
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
