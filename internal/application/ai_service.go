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

func (s *AIService) Models() []domain.ModelInfo {
	models := make([]domain.ModelInfo, 0, len(s.models))
	for i, model := range s.models {
		provider, name := splitModel(model)
		models = append(models, domain.ModelInfo{
			Name:     model,
			Provider: provider,
			Model:    name,
			Order:    i + 1,
		})
	}
	return models
}

func (s *AIService) CheckModels(ctx context.Context, models []string) []domain.ModelCheck {
	candidates := normalizeModels(models)
	if len(candidates) == 0 {
		candidates = append([]string(nil), s.models...)
	}

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
		return []string{strings.TrimSpace(req.Model)}, nil
	}
	if models := normalizeModels(req.Models); len(models) > 0 {
		return models, nil
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

func splitModel(candidate string) (string, string) {
	if provider, model, ok := strings.Cut(candidate, ":"); ok {
		return strings.TrimSpace(provider), strings.TrimSpace(model)
	}
	return "", strings.TrimSpace(candidate)
}
