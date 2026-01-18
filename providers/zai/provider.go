package zai

import (
	"bufio"
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/ulgerang/llm-module/llm"
	"github.com/ulgerang/llm-module/logger"
	"github.com/ulgerang/llm-module/utils"
)

const (
	defaultModel   = "glm-4.7"
	defaultBaseURL = "https://api.z.ai/api/coding/paas/v4"
)

// Provider implements llm.Provider for Z.AI models using direct HTTP calls.
type Provider struct {
	httpClient *http.Client
	apiKey     string
	baseURL    string
	logger     logger.Logger
	modelName  string
}

// ChatRequest represents the Z.AI chat completion request.
type ChatRequest struct {
	Model          string          `json:"model"`
	Messages       []ChatMessage   `json:"messages"`
	Stream         bool            `json:"stream,omitempty"`
	Temperature    *float64        `json:"temperature,omitempty"`
	TopP           *float64        `json:"top_p,omitempty"`
	MaxTokens      *int64          `json:"max_tokens,omitempty"`
	DoSample       *bool           `json:"do_sample,omitempty"`
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`
	Thinking       *ThinkingConfig `json:"thinking,omitempty"`
}

// ThinkingConfig specifies reasoning capabilities.
type ThinkingConfig struct {
	Type string `json:"type"`
}

// ResponseFormat specifies the format of the response.
type ResponseFormat struct {
	Type string `json:"type"`
}

// ChatMessage represents a message in the chat.
type ChatMessage struct {
	Role             string `json:"role"`
	Content          string `json:"content"`
	ReasoningContent string `json:"reasoning_content,omitempty"`
}

// ChatResponse represents the Z.AI chat completion response.
type ChatResponse struct {
	ID        string       `json:"id"`
	RequestID string       `json:"request_id"`
	Created   int64        `json:"created"`
	Model     string       `json:"model"`
	Choices   []ChatChoice `json:"choices"`
	Usage     Usage        `json:"usage"`
}

// ChatChoice represents a choice in the response.
type ChatChoice struct {
	Index        int         `json:"index"`
	Message      ChatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

// StreamChunkResponse represents a streaming chunk from Z.AI.
type StreamChunkResponse struct {
	ID      string              `json:"id"`
	Created int64               `json:"created"`
	Model   string              `json:"model"`
	Choices []StreamChunkChoice `json:"choices"`
	Usage   *Usage              `json:"usage,omitempty"`
}

// StreamChunkChoice represents a streaming choice.
type StreamChunkChoice struct {
	Index        int         `json:"index"`
	Delta        StreamDelta `json:"delta"`
	FinishReason string      `json:"finish_reason,omitempty"`
}

// StreamDelta represents the delta content in streaming.
type StreamDelta struct {
	Role             string `json:"role,omitempty"`
	Content          string `json:"content,omitempty"`
	ReasoningContent string `json:"reasoning_content,omitempty"`
}

// Usage represents token usage information.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ErrorResponse represents an error from Z.AI API.
type ErrorResponse struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// New creates a new Z.AI provider instance using the default coding endpoint.
func New(log logger.Logger, apiKey, modelName string) (*Provider, error) {
	return newProvider(log, apiKey, modelName, defaultBaseURL)
}

// NewWithBaseURL creates a new provider using a custom base URL.
func NewWithBaseURL(log logger.Logger, apiKey, modelName, baseURL string) (*Provider, error) {
	return newProvider(log, apiKey, modelName, baseURL)
}

func newProvider(log logger.Logger, apiKey, modelName, baseURL string) (*Provider, error) {
	if apiKey == "" {
		apiKey = os.Getenv("ZAI_API_KEY")
		if apiKey == "" {
			return nil, errors.New("ZAI_API_KEY not provided")
		}
	}

	if modelName == "" {
		modelName = os.Getenv("ZAI_MODEL")
		if modelName == "" {
			modelName = defaultModel
		}
	}

	if baseURL == "" {
		baseURL = defaultBaseURL
	}

	// Create HTTP client with generous timeouts for LLM API calls
	// These APIs can be slow, especially for complex generation tasks
	transport := &http.Transport{
		DialContext: (&net.Dialer{
			Timeout:   30 * time.Second, // Connection timeout
			KeepAlive: 30 * time.Second,
		}).DialContext,
		TLSHandshakeTimeout:   30 * time.Second,  // TLS handshake timeout (was 10s default)
		ResponseHeaderTimeout: 300 * time.Second, // Wait for response headers (5 min)
		ExpectContinueTimeout: 1 * time.Second,
		IdleConnTimeout:       90 * time.Second,
		MaxIdleConns:          100,
		MaxIdleConnsPerHost:   10,
		TLSClientConfig: &tls.Config{
			MinVersion: tls.VersionTLS12,
		},
	}

	httpClient := &http.Client{
		Transport: transport,
		Timeout:   600 * time.Second, // Overall request timeout (10 min for long generation)
	}

	return &Provider{
		httpClient: httpClient,
		apiKey:     apiKey,
		baseURL:    baseURL,
		logger:     log,
		modelName:  modelName,
	}, nil
}

// GetModelName returns the configured model name.
func (p *Provider) GetModelName() string {
	return p.modelName
}

// GenerateText performs a non-streaming Z.AI request.
func (p *Provider) GenerateText(ctx context.Context, prompt string, opts ...llm.GenerationOption) (string, *llm.UsageInfo, error) {
	options := &llm.GenerationOptions{}
	for _, opt := range opts {
		opt(options)
	}

	p.logger.Debug(fmt.Sprintf("[ZAI] Sending request to model: %s", p.modelName))

	systemPrompt := p.composeSystemPrompt(options)

	messages := []ChatMessage{}
	if systemPrompt != "" {
		messages = append(messages, ChatMessage{Role: "system", Content: systemPrompt})
	}
	messages = append(messages, ChatMessage{Role: "user", Content: prompt})

	req := ChatRequest{
		Model:    p.modelName,
		Messages: messages,
		// Stream:   false,  // Removed for ZAI API compatibility
	}

	if options.ResponseSchema != nil || strings.Contains(strings.ToLower(options.ResponseFormat), "json") {
		req.ResponseFormat = &ResponseFormat{Type: "json_object"}
	}

	if options.Temperature != nil {
		temp := float64(*options.Temperature)
		req.Temperature = &temp
	}
	if options.MaxTokens != nil {
		maxTok := int64(*options.MaxTokens)
		req.MaxTokens = &maxTok
	}
	if options.TopP != nil {
		topP := float64(*options.TopP)
		req.TopP = &topP
	}

	// Enable thinking for GLM-4.7 models by default
	// Thinking mode uses reasoning tokens WITHIN max_tokens budget, so we must ensure
	// sufficient tokens for BOTH reasoning (500-2000+) AND actual content output.
	// BUG FIX: Increased from 8192 to 16384 to prevent reasoning consuming entire budget.
	if strings.Contains(p.modelName, "glm-4.7") {
		req.Thinking = &ThinkingConfig{Type: "enabled"}
		// Ensure minimum tokens for thinking mode - reasoning consumes part of the budget
		minTokens := int64(16384)
		if req.MaxTokens == nil || *req.MaxTokens < minTokens {
			req.MaxTokens = &minTokens
		}
	}

	body, err := json.Marshal(req)
	if err != nil {
		return "", nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return "", nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		p.logger.Error("[ZAI] Failed to send request", err)
		return "", nil, err
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		var errResp ErrorResponse
		if json.Unmarshal(respBody, &errResp) == nil && (errResp.Code != "" || errResp.Message != "") {
			return "", nil, fmt.Errorf("Z.AI API error (code %s): %s", errResp.Code, errResp.Message)
		}

		// Check for nested error object (standard OpenAI format)
		var wrappedResp struct {
			Error struct {
				Code    interface{} `json:"code"`
				Message string      `json:"message"`
			} `json:"error"`
		}
		if json.Unmarshal(respBody, &wrappedResp) == nil && wrappedResp.Error.Message != "" {
			return "", nil, fmt.Errorf("Z.AI API error (code %v): %s", wrappedResp.Error.Code, wrappedResp.Error.Message)
		}

		return "", nil, fmt.Errorf("Z.AI API error: %s", string(respBody))
	}

	var chatResp ChatResponse
	if err := json.Unmarshal(respBody, &chatResp); err != nil {
		return "", nil, fmt.Errorf("failed to parse response: %w", err)
	}

	if len(chatResp.Choices) == 0 {
		p.logger.Warning("[ZAI] No content generated")
		return "", nil, errors.New("no content generated")
	}

	message := chatResp.Choices[0].Message
	generated := message.Content
	// BUG FIX: Do NOT fall back to reasoning_content when content is empty.
	// This was causing LLM "thinking" to be returned as actual output.
	// If content is empty but reasoning exists, the token budget was likely exhausted.
	if generated == "" && message.ReasoningContent != "" {
		p.logger.Warning("[ZAI] Content is empty but reasoning_content exists - token budget may be insufficient")
		return "", nil, errors.New("no content generated: reasoning consumed entire token budget, increase max_tokens")
	}
	if generated == "" {
		p.logger.Warning("[ZAI] No content generated")
		return "", nil, errors.New("no content generated")
	}
	if options.ResponseSchema != nil {
		if extracted, extractErr := utils.ExtractJSONFromString(generated); extractErr == nil {
			generated = extracted
		} else {
			p.logger.Warningf("[ZAI] Failed to extract JSON: %v", extractErr)
		}
	}

	usage := &llm.UsageInfo{
		InputTokens:  chatResp.Usage.PromptTokens,
		OutputTokens: chatResp.Usage.CompletionTokens,
	}

	p.logger.Debug(fmt.Sprintf("Generated text (ZAI/%s): %s", p.modelName, generated))
	return generated, usage, nil
}

// GenerateTextStream streams responses from Z.AI.
func (p *Provider) GenerateTextStream(ctx context.Context, prompt string, outChan chan<- llm.StreamChunk, opts ...llm.GenerationOption) (*llm.UsageInfo, error) {
	defer close(outChan)

	options := &llm.GenerationOptions{
		Temperature: llm.ValuePtr(float32(0.7)),
		MaxTokens:   llm.ValuePtr(int32(4096)),
		System:      "You are a helpful assistant.",
	}
	for _, opt := range opts {
		opt(options)
	}

	systemPrompt := p.composeSystemPrompt(options)

	messages := []ChatMessage{}
	if systemPrompt != "" {
		messages = append(messages, ChatMessage{Role: "system", Content: systemPrompt})
	}
	messages = append(messages, ChatMessage{Role: "user", Content: prompt})

	req := ChatRequest{
		Model:    p.modelName,
		Messages: messages,
		Stream:   true, // Enable streaming for ZAI API
	}

	if options.ResponseSchema != nil || strings.Contains(strings.ToLower(options.ResponseFormat), "json") {
		req.ResponseFormat = &ResponseFormat{Type: "json_object"}
	}

	if options.Temperature != nil {
		temp := float64(*options.Temperature)
		req.Temperature = &temp
	}
	if options.MaxTokens != nil {
		maxTok := int64(*options.MaxTokens)
		req.MaxTokens = &maxTok
	}
	if options.TopP != nil {
		topP := float64(*options.TopP)
		req.TopP = &topP
	}

	// Enable thinking for GLM-4.7 models by default
	// Thinking mode uses reasoning tokens WITHIN max_tokens budget, so we must ensure
	// sufficient tokens for BOTH reasoning (500-2000+) AND actual content output.
	// BUG FIX: Increased from 8192 to 16384 to prevent reasoning consuming entire budget.
	if strings.Contains(p.modelName, "glm-4.7") {
		req.Thinking = &ThinkingConfig{Type: "enabled"}
		// Ensure minimum tokens for thinking mode - reasoning consumes part of the budget
		minTokens := int64(16384)
		if req.MaxTokens == nil || *req.MaxTokens < minTokens {
			req.MaxTokens = &minTokens
		}
	}

	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		p.logger.Error("[ZAI] Failed to send streaming request", err)
		outChan <- llm.StreamChunk{Err: err}
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		var errResp ErrorResponse
		if json.Unmarshal(respBody, &errResp) == nil && (errResp.Code != "" || errResp.Message != "") {
			err := fmt.Errorf("Z.AI API error (code %s): %s", errResp.Code, errResp.Message)
			outChan <- llm.StreamChunk{Err: err}
			return nil, err
		}
		// Check for nested error object
		var wrappedResp struct {
			Error struct {
				Code    interface{} `json:"code"`
				Message string      `json:"message"`
			} `json:"error"`
		}
		if json.Unmarshal(respBody, &wrappedResp) == nil && wrappedResp.Error.Message != "" {
			err := fmt.Errorf("Z.AI API error (code %v): %s", wrappedResp.Error.Code, wrappedResp.Error.Message)
			outChan <- llm.StreamChunk{Err: err}
			return nil, err
		}

		err := fmt.Errorf("Z.AI API error: %s", string(respBody))
		outChan <- llm.StreamChunk{Err: err}
		return nil, err
	}

	var usage *llm.UsageInfo
	scanner := bufio.NewScanner(resp.Body)
	// Increase buffer size to handle large responses (default 64K -> 10MB)
	const maxTokenSize = 10 * 1024 * 1024 // 10MB
	buf := make([]byte, maxTokenSize)
	scanner.Buffer(buf, maxTokenSize)
	for scanner.Scan() {
		line := scanner.Text()

		// Skip empty lines
		if line == "" {
			continue
		}

		// Check for stream end
		if line == "data: [DONE]" {
			break
		}

		// Parse SSE data
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")
		var chunk StreamChunkResponse
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			p.logger.Warningf("[ZAI] Failed to parse streaming chunk: %v", err)
			continue
		}

		if len(chunk.Choices) > 0 {
			delta := chunk.Choices[0].Delta
			// BUG FIX: Only output actual content, NOT reasoning_content.
			// ZAI's GLM model sends reasoning_content first, then content.
			// Previously we were outputting both, which leaked LLM "thinking" to output.
			content := delta.Content
			// Skip reasoning_content chunks - they should not appear in output
			if content != "" {
				select {
				case outChan <- llm.StreamChunk{Delta: content}:
				case <-ctx.Done():
					p.logger.Info("[ZAI] Context cancelled during stream send")
					return nil, ctx.Err()
				}
			}
		}

		// Capture usage from the last chunk
		if chunk.Usage != nil {
			usage = &llm.UsageInfo{
				InputTokens:  chunk.Usage.PromptTokens,
				OutputTokens: chunk.Usage.CompletionTokens,
			}
		}
	}

	if err := scanner.Err(); err != nil {
		p.logger.Error("[ZAI] Stream scanner error", err)
		outChan <- llm.StreamChunk{Err: err}
		return nil, err
	}

	outChan <- llm.StreamChunk{IsFinal: true}
	return usage, nil
}

// Close releases resources.
func (p *Provider) Close() error {
	p.logger.Info("[ZAI] Provider closed.")
	return nil
}

func (p *Provider) composeSystemPrompt(options *llm.GenerationOptions) string {
	var builder strings.Builder

	if len(options.SystemBlocks) > 0 {
		for _, block := range options.SystemBlocks {
			builder.WriteString(block.Text)
			builder.WriteString("\n\n")
		}
	}

	if options.System != "" {
		builder.WriteString(options.System)
		builder.WriteString("\n\n")
	}

	if options.Language != "" {
		builder.WriteString(fmt.Sprintf("Please respond in %s language.", utils.GetLangName(options.Language)))
		builder.WriteString("\n\n")
	}

	if options.ResponseSchema != nil {
		p.logger.Debugf("[ZAI] ResponseSchema received: %+v", options.ResponseSchema)
		schemaJSON, err := llm.ConvertToJSONSchema(options.ResponseSchema)
		if err != nil {
			p.logger.Warningf("[ZAI] Failed to convert response schema: %v", err)
		} else {
			p.logger.Infof("[ZAI] ResponseSchema applied successfully, schema length: %d chars", len(schemaJSON))
			builder.WriteString("Please provide your response strictly in the following JSON format:\n")
			builder.WriteString(schemaJSON)
			builder.WriteString("\n\n")
		}
	}

	if options.ResponseFormat != "" {
		builder.WriteString(fmt.Sprintf("Response format: %s\n\n", options.ResponseFormat))

		// BUG-004 fix: When JSON format is requested but no schema provided,
		// add strict instructions to prevent LLM from adding explanatory text
		if strings.Contains(strings.ToLower(options.ResponseFormat), "json") && options.ResponseSchema == nil {
			builder.WriteString("\n!!! CRITICAL - JSON OUTPUT RULES !!!\n")
			builder.WriteString("You MUST output ONLY valid JSON. Follow these rules strictly:\n")
			builder.WriteString("1. Start your response with the '{' character\n")
			builder.WriteString("2. End your response with the '}' character\n")
			builder.WriteString("3. Do NOT include any text before or after the JSON\n")
			builder.WriteString("4. Do NOT use markdown code blocks\n")
			builder.WriteString("5. Do NOT add explanations, comments, or descriptions\n\n")
		}
	}

	return strings.TrimSpace(builder.String())
}
