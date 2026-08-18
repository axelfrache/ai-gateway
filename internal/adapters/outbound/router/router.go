package router

import (
	"context"
	"fmt"
	"strings"

	"github.com/frachea/ai-gateway/internal/domain"
)

type Router struct {
	defaultProvider string
	providers       map[string]domain.AIProvider
}

func New(defaultProvider string, providers map[string]domain.AIProvider) *Router {
	copyProviders := make(map[string]domain.AIProvider, len(providers))
	for name, provider := range providers {
		name = strings.TrimSpace(strings.ToLower(name))
		if name == "" || provider == nil {
			continue
		}
		copyProviders[name] = provider
	}

	return &Router{
		defaultProvider: strings.TrimSpace(strings.ToLower(defaultProvider)),
		providers:       copyProviders,
	}
}

func (r *Router) Generate(ctx context.Context, candidate string, req domain.GenerateRequest) (domain.GenerateResponse, error) {
	providerName, model := splitCandidate(candidate, r.defaultProvider)
	provider := r.providers[providerName]
	if provider == nil {
		return domain.GenerateResponse{}, domain.NewError(
			domain.ErrorKindModelUnavailable,
			0,
			fmt.Sprintf("provider %q is not configured", providerName),
			nil,
		)
	}

	response, err := provider.Generate(ctx, model, req)
	if err != nil {
		return domain.GenerateResponse{}, err
	}
	if providerName != "" && !strings.HasPrefix(response.Model, providerName+":") {
		response.Model = providerName + ":" + response.Model
	}
	return response, nil
}

func splitCandidate(candidate, defaultProvider string) (string, string) {
	candidate = strings.TrimSpace(candidate)
	providerName := defaultProvider
	model := candidate

	if before, after, ok := strings.Cut(candidate, ":"); ok {
		providerName = strings.TrimSpace(strings.ToLower(before))
		model = strings.TrimSpace(after)
	}

	return providerName, model
}
