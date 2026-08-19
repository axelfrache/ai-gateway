package main

import (
	"context"
	"log"
	"net/http"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/frachea/ai-gateway/internal/adapters/inbound/httpapi"
	"github.com/frachea/ai-gateway/internal/adapters/outbound/gemini"
	"github.com/frachea/ai-gateway/internal/adapters/outbound/openai"
	"github.com/frachea/ai-gateway/internal/adapters/outbound/router"
	"github.com/frachea/ai-gateway/internal/application"
	"github.com/frachea/ai-gateway/internal/config"
	"github.com/frachea/ai-gateway/internal/domain"
	"github.com/frachea/ai-gateway/internal/platform/logger"
)

func main() {
	if err := config.LoadDotEnv(".env"); err != nil {
		log.Fatalf("load .env: %v", err)
	}

	cfg, err := config.FromEnv()
	if err != nil {
		log.Fatalf("load config: %v", err)
	}

	logr := logger.New()
	providers := map[string]domain.AIProvider{}
	if cfg.GeminiAPIKey != "" {
		providers["gemini"] = gemini.NewClient(cfg.GeminiAPIKey, cfg.GeminiBaseURL, cfg.RequestTimeout)
	}
	if cfg.GroqAPIKey != "" {
		providers["groq"] = openai.NewClient("Groq", cfg.GroqAPIKey, cfg.GroqBaseURL, "max_completion_tokens", cfg.RequestTimeout)
	}
	if cfg.MistralAPIKey != "" {
		providers["mistral"] = openai.NewClient("Mistral", cfg.MistralAPIKey, cfg.MistralBaseURL, "max_tokens", cfg.RequestTimeout)
	}
	if cfg.OpenRouterAPIKey != "" {
		providers["openrouter"] = openai.NewOpenRouterClient(cfg.OpenRouterAPIKey, cfg.OpenRouterBaseURL, cfg.RequestTimeout)
	}

	modelFallbacks := configuredFallbacks(cfg.ModelFallbacks, providers)
	aiRouter := router.New(defaultProvider(providers), providers)
	aiService, err := application.NewAIService(aiRouter, modelFallbacks)
	if err != nil {
		log.Fatalf("create AI service: %v", err)
	}

	server := &http.Server{
		Addr:         cfg.ServerAddr,
		Handler:      httpapi.NewServer(aiService, logr, cfg.GatewayAPIKeys).Routes(),
		ReadTimeout:  10 * time.Second,
		WriteTimeout: cfg.RequestTimeout + 5*time.Second,
		IdleTimeout:  60 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	go func() {
		logr.Info("starting API", "addr", cfg.ServerAddr, "models", modelFallbacks)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("serve: %v", err)
		}
	}()

	<-ctx.Done()

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := server.Shutdown(shutdownCtx); err != nil {
		log.Fatalf("shutdown: %v", err)
	}
}

func configuredFallbacks(candidates []string, providers map[string]domain.AIProvider) []string {
	fallbacks := make([]string, 0, len(candidates))
	for _, candidate := range candidates {
		provider := providerName(candidate)
		if provider == "" {
			provider = defaultProvider(providers)
		}
		if providers[provider] != nil {
			fallbacks = append(fallbacks, candidate)
		}
	}
	return fallbacks
}

func providerName(candidate string) string {
	before, _, ok := strings.Cut(strings.TrimSpace(candidate), ":")
	if !ok {
		return ""
	}
	return strings.ToLower(strings.TrimSpace(before))
}

func defaultProvider(providers map[string]domain.AIProvider) string {
	for _, name := range []string{"gemini", "groq", "mistral", "openrouter"} {
		if providers[name] != nil {
			return name
		}
	}
	return ""
}
