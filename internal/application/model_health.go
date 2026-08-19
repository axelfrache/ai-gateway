package application

import (
	"sync"
	"time"

	"github.com/frachea/ai-gateway/internal/domain"
)

// cooldownByKind controls how long a model is deprioritized after a
// retryable failure. Kinds absent here (invalid_request, auth, safety, ...)
// are never cached: they describe the request or credentials, not a
// transient model outage, so caching them would not help future requests.
var cooldownByKind = map[domain.ErrorKind]time.Duration{
	domain.ErrorKindRateLimited:      30 * time.Second,
	domain.ErrorKindTemporary:        15 * time.Second,
	domain.ErrorKindModelUnavailable: 2 * time.Minute,
}

// modelHealth remembers, per "provider:model" candidate, whether its last
// outcome was a retryable failure. It never removes a model from a fallback
// chain outright — it only reorders candidates so a request tries recently
// healthy models before recently failing ones, cutting the number of dead
// attempts per request without weakening the resilience guarantee that every
// configured model is still eventually tried.
type modelHealth struct {
	mu      sync.RWMutex
	blocked map[string]time.Time
	now     func() time.Time
}

func newModelHealth() *modelHealth {
	return &modelHealth{
		blocked: map[string]time.Time{},
		now:     time.Now,
	}
}

func (h *modelHealth) recordSuccess(model string) {
	h.mu.Lock()
	defer h.mu.Unlock()
	delete(h.blocked, model)
}

func (h *modelHealth) recordFailure(model string, kind domain.ErrorKind) {
	cooldown, ok := cooldownByKind[kind]
	if !ok {
		return
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	h.blocked[model] = h.now().Add(cooldown)
}

func (h *modelHealth) available(model string) bool {
	h.mu.RLock()
	until, blocked := h.blocked[model]
	h.mu.RUnlock()
	if !blocked {
		return true
	}
	return !h.now().Before(until)
}

// order returns models with any currently-cooling-down candidates moved to
// the end, preserving relative order within each group.
func (h *modelHealth) order(models []string) []string {
	if len(models) <= 1 {
		return models
	}

	available := make([]string, 0, len(models))
	blocked := make([]string, 0, len(models))
	for _, model := range models {
		if h.available(model) {
			available = append(available, model)
		} else {
			blocked = append(blocked, model)
		}
	}
	return append(available, blocked...)
}
