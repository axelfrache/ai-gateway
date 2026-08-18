package domain

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
)

type GenerateRequest struct {
	Prompt          string
	System          string
	Temperature     *float64
	MaxOutputTokens *int
	ResponseSchema  json.RawMessage
}

type GenerateResponse struct {
	Model string
	Text  string
}

type Attempt struct {
	Model         string `json:"model"`
	Status        string `json:"status"`
	Error         string `json:"error,omitempty"`
	LatencyMillis int64  `json:"latency_ms"`
}

type GenerateResult struct {
	Model        string    `json:"model"`
	Text         string    `json:"text"`
	FallbackUsed bool      `json:"fallback_used"`
	Attempts     []Attempt `json:"attempts"`
}

type AIProvider interface {
	Generate(ctx context.Context, model string, req GenerateRequest) (GenerateResponse, error)
}

type ErrorKind string

const (
	ErrorKindInvalidRequest   ErrorKind = "invalid_request"
	ErrorKindAuth             ErrorKind = "auth"
	ErrorKindRateLimited      ErrorKind = "rate_limited"
	ErrorKindTemporary        ErrorKind = "temporary"
	ErrorKindModelUnavailable ErrorKind = "model_unavailable"
	ErrorKindSafety           ErrorKind = "safety"
	ErrorKindNoModel          ErrorKind = "no_model_available"
	ErrorKindUnknown          ErrorKind = "unknown"
)

type AIError struct {
	Kind       ErrorKind
	StatusCode int
	Message    string
	Err        error
}

func (e *AIError) Error() string {
	if e == nil {
		return ""
	}
	if e.Message != "" {
		return e.Message
	}
	if e.Err != nil {
		return e.Err.Error()
	}
	return string(e.Kind)
}

func (e *AIError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

func NewError(kind ErrorKind, statusCode int, message string, err error) *AIError {
	return &AIError{
		Kind:       kind,
		StatusCode: statusCode,
		Message:    message,
		Err:        err,
	}
}

func Retryable(err error) bool {
	var aiErr *AIError
	if !errors.As(err, &aiErr) {
		return false
	}
	return aiErr.Kind == ErrorKindRateLimited ||
		aiErr.Kind == ErrorKindTemporary ||
		aiErr.Kind == ErrorKindModelUnavailable
}

func Kind(err error) ErrorKind {
	var aiErr *AIError
	if !errors.As(err, &aiErr) {
		return ErrorKindUnknown
	}
	return aiErr.Kind
}

func NoModelAvailable(attempts int) *AIError {
	return NewError(
		ErrorKindNoModel,
		0,
		fmt.Sprintf("no AI model produced a response after %d attempt(s)", attempts),
		nil,
	)
}
