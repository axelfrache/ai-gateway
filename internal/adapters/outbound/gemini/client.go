package gemini

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/frachea/ai-gateway/internal/domain"
)

type Client struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
}

func NewClient(apiKey, baseURL string, timeout time.Duration) *Client {
	return &Client{
		apiKey:  apiKey,
		baseURL: strings.TrimRight(baseURL, "/"),
		httpClient: &http.Client{
			Timeout: timeout,
		},
	}
}

func (c *Client) Generate(ctx context.Context, model string, req domain.GenerateRequest) (domain.GenerateResponse, error) {
	payload := generateContentRequest{
		Contents: []content{
			{
				Role: "user",
				Parts: []part{
					{Text: req.Prompt},
				},
			},
		},
	}

	if strings.TrimSpace(req.System) != "" {
		payload.SystemInstruction = &content{
			Parts: []part{{Text: req.System}},
		}
	}

	if req.Temperature != nil || req.MaxOutputTokens != nil || len(req.ResponseSchema) > 0 {
		payload.GenerationConfig = &generationConfig{
			Temperature:     req.Temperature,
			MaxOutputTokens: req.MaxOutputTokens,
		}
		if len(req.ResponseSchema) > 0 {
			responseSchema, err := normalizeSchema(req.ResponseSchema)
			if err != nil {
				return domain.GenerateResponse{}, domain.NewError(domain.ErrorKindInvalidRequest, 0, "failed to normalize Gemini response schema", err)
			}
			payload.GenerationConfig.ResponseMimeType = "application/json"
			payload.GenerationConfig.ResponseSchema = responseSchema
		}
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return domain.GenerateResponse{}, domain.NewError(domain.ErrorKindInvalidRequest, 0, "failed to encode Gemini request", err)
	}

	endpoint := fmt.Sprintf("%s/models/%s:generateContent?key=%s", c.baseURL, url.PathEscape(model), url.QueryEscape(c.apiKey))
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return domain.GenerateResponse{}, domain.NewError(domain.ErrorKindInvalidRequest, 0, "failed to build Gemini request", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	httpResp, err := c.httpClient.Do(httpReq)
	if err != nil {
		if errors.Is(err, context.Canceled) {
			return domain.GenerateResponse{}, domain.NewError(domain.ErrorKindTemporary, 0, "request was canceled", err)
		}
		if errors.Is(err, context.DeadlineExceeded) {
			return domain.GenerateResponse{}, domain.NewError(domain.ErrorKindTemporary, 0, "request timed out", err)
		}
		return domain.GenerateResponse{}, domain.NewError(domain.ErrorKindTemporary, 0, "Gemini request failed", err)
	}
	defer httpResp.Body.Close()

	responseBody, err := io.ReadAll(io.LimitReader(httpResp.Body, 4<<20))
	if err != nil {
		return domain.GenerateResponse{}, domain.NewError(domain.ErrorKindTemporary, httpResp.StatusCode, "failed to read Gemini response", err)
	}

	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		return domain.GenerateResponse{}, classifyGeminiError(httpResp.StatusCode, responseBody)
	}

	var geminiResp generateContentResponse
	if err := json.Unmarshal(responseBody, &geminiResp); err != nil {
		return domain.GenerateResponse{}, domain.NewError(domain.ErrorKindTemporary, httpResp.StatusCode, "failed to decode Gemini response", err)
	}

	text := geminiResp.Text()
	if text == "" {
		if geminiResp.PromptFeedback.BlockReason != "" {
			return domain.GenerateResponse{}, domain.NewError(domain.ErrorKindSafety, httpResp.StatusCode, "Gemini blocked the prompt: "+geminiResp.PromptFeedback.BlockReason, nil)
		}
		if finishReason := geminiResp.FirstFinishReason(); finishReason == "SAFETY" || finishReason == "RECITATION" {
			return domain.GenerateResponse{}, domain.NewError(domain.ErrorKindSafety, httpResp.StatusCode, "Gemini stopped generation: "+finishReason, nil)
		}
		return domain.GenerateResponse{}, domain.NewError(domain.ErrorKindTemporary, httpResp.StatusCode, "Gemini returned no text", nil)
	}

	return domain.GenerateResponse{
		Model: model,
		Text:  text,
	}, nil
}

type generateContentRequest struct {
	Contents          []content         `json:"contents"`
	SystemInstruction *content          `json:"systemInstruction,omitempty"`
	GenerationConfig  *generationConfig `json:"generationConfig,omitempty"`
}

type content struct {
	Role  string `json:"role,omitempty"`
	Parts []part `json:"parts"`
}

type part struct {
	Text string `json:"text"`
}

type generationConfig struct {
	Temperature      *float64        `json:"temperature,omitempty"`
	MaxOutputTokens  *int            `json:"maxOutputTokens,omitempty"`
	ResponseMimeType string          `json:"responseMimeType,omitempty"`
	ResponseSchema   json.RawMessage `json:"responseSchema,omitempty"`
}

type generateContentResponse struct {
	Candidates []struct {
		Content struct {
			Parts []struct {
				Text string `json:"text"`
			} `json:"parts"`
		} `json:"content"`
		FinishReason string `json:"finishReason"`
	} `json:"candidates"`
	PromptFeedback struct {
		BlockReason string `json:"blockReason"`
	} `json:"promptFeedback"`
}

func (r generateContentResponse) Text() string {
	var builder strings.Builder
	for _, candidate := range r.Candidates {
		for _, part := range candidate.Content.Parts {
			if part.Text == "" {
				continue
			}
			if builder.Len() > 0 {
				builder.WriteString("\n")
			}
			builder.WriteString(part.Text)
		}
		if builder.Len() > 0 {
			break
		}
	}
	return strings.TrimSpace(builder.String())
}

func (r generateContentResponse) FirstFinishReason() string {
	for _, candidate := range r.Candidates {
		if candidate.FinishReason != "" {
			return candidate.FinishReason
		}
	}
	return ""
}

func normalizeSchema(schema json.RawMessage) (json.RawMessage, error) {
	var value any
	if err := json.Unmarshal(schema, &value); err != nil {
		return nil, err
	}
	removeAdditionalProperties(value)

	normalized, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	return normalized, nil
}

func removeAdditionalProperties(value any) {
	switch typed := value.(type) {
	case map[string]any:
		delete(typed, "additionalProperties")
		for _, child := range typed {
			removeAdditionalProperties(child)
		}
	case []any:
		for _, child := range typed {
			removeAdditionalProperties(child)
		}
	}
}

type errorResponse struct {
	Error struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
		Status  string `json:"status"`
	} `json:"error"`
}

func classifyGeminiError(statusCode int, body []byte) error {
	var parsed errorResponse
	_ = json.Unmarshal(body, &parsed)

	message := strings.TrimSpace(parsed.Error.Message)
	if message == "" {
		message = http.StatusText(statusCode)
	}
	if message == "" {
		message = fmt.Sprintf("Gemini returned HTTP %d", statusCode)
	}

	status := strings.ToUpper(parsed.Error.Status)

	switch {
	case statusCode == http.StatusTooManyRequests || status == "RESOURCE_EXHAUSTED":
		return domain.NewError(domain.ErrorKindRateLimited, statusCode, message, nil)
	case statusCode == http.StatusUnauthorized || statusCode == http.StatusForbidden:
		return domain.NewError(domain.ErrorKindAuth, statusCode, message, nil)
	case statusCode == http.StatusBadRequest || status == "INVALID_ARGUMENT":
		return domain.NewError(domain.ErrorKindInvalidRequest, statusCode, message, nil)
	case statusCode == http.StatusNotFound || status == "NOT_FOUND":
		return domain.NewError(domain.ErrorKindModelUnavailable, statusCode, message, nil)
	case statusCode >= 500:
		return domain.NewError(domain.ErrorKindTemporary, statusCode, message, nil)
	default:
		return domain.NewError(domain.ErrorKindUnknown, statusCode, message, nil)
	}
}
