package application

import (
	"testing"
	"time"

	"github.com/frachea/ai-gateway/internal/domain"
)

func TestModelHealthOrdersBlockedModelsLast(t *testing.T) {
	health := newModelHealth()
	health.recordFailure("gemini:gemini-3.6-flash", domain.ErrorKindRateLimited)

	ordered := health.order([]string{"gemini:gemini-3.6-flash", "groq:openai/gpt-oss-120b"})
	if ordered[0] != "groq:openai/gpt-oss-120b" || ordered[1] != "gemini:gemini-3.6-flash" {
		t.Fatalf("expected blocked model last, got %#v", ordered)
	}
}

func TestModelHealthRecoversAfterCooldown(t *testing.T) {
	current := time.Now()
	health := newModelHealth()
	health.now = func() time.Time { return current }
	health.recordFailure("gemini:gemini-3.6-flash", domain.ErrorKindTemporary)

	ordered := health.order([]string{"gemini:gemini-3.6-flash", "groq:openai/gpt-oss-120b"})
	if ordered[0] != "groq:openai/gpt-oss-120b" {
		t.Fatalf("expected model still in cooldown, got %#v", ordered)
	}

	current = current.Add(cooldownByKind[domain.ErrorKindTemporary] + time.Second)
	ordered = health.order([]string{"gemini:gemini-3.6-flash", "groq:openai/gpt-oss-120b"})
	if ordered[0] != "gemini:gemini-3.6-flash" {
		t.Fatalf("expected model available again after cooldown expired, got %#v", ordered)
	}
}

func TestModelHealthRecordSuccessClearsCooldown(t *testing.T) {
	health := newModelHealth()
	health.recordFailure("gemini:gemini-3.6-flash", domain.ErrorKindRateLimited)
	health.recordSuccess("gemini:gemini-3.6-flash")

	ordered := health.order([]string{"gemini:gemini-3.6-flash", "groq:openai/gpt-oss-120b"})
	if ordered[0] != "gemini:gemini-3.6-flash" {
		t.Fatalf("expected model available again after success, got %#v", ordered)
	}
}

func TestModelHealthIgnoresNonRetryableKinds(t *testing.T) {
	health := newModelHealth()
	health.recordFailure("gemini:gemini-3.6-flash", domain.ErrorKindInvalidRequest)

	ordered := health.order([]string{"gemini:gemini-3.6-flash", "groq:openai/gpt-oss-120b"})
	if ordered[0] != "gemini:gemini-3.6-flash" {
		t.Fatalf("expected non-retryable kind to not affect ordering, got %#v", ordered)
	}
}
