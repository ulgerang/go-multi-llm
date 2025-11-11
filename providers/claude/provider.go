package claude

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/ulgerang/llm-module/llm"
	"github.com/ulgerang/llm-module/logger"
	"github.com/ulgerang/llm-module/utils"
)

const (
	defaultClaudeModel   = "claude-opus-4-20250514"
	defaultClaudeBaseURL = "https://api.anthropic.com/v1"
	claudeAPIVersion     = "2023-06-01"
	defaultClaudeTimeout = 60 * time.Second
)

// Provider implements llm.Provider for Anthropic Claude.
type Provider struct {
	client    *http.Client
	logger    logger.Logger
	apiKey    string
	modelName string
	baseURL   string
}

// StreamEvent represents a single event in the Claude SSE stream.
type StreamEvent struct {
	Type         string             `json:"type"`
	Index        *int               `json:"index,omitempty"`
	Delta        *StreamDelta       `json:"delta,omitempty"`
	Message      *MessageResponse   `json:"message,omitempty"`
	Usage        *Usage             `json:"usage,omitempty"`
	ContentBlock *ContentBlock      `json:"content_block,omitempty"`
	Error        *ErrorDetail       `json:"error,omitempty"`
}

// ErrorDetail captures Claude stream error information.
type ErrorDetail struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

// StreamDelta represents incremental text payloads.
type StreamDelta struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// ToolInputSchema defines Claude tool schema payload.
type ToolInputSchema struct {
	Type        string                            `json:"type"`
	Properties  map[string]map[string]interface{} `json:"properties"`
	Required    []string                          `json:"required,omitempty"`
	Description string                            `json:"description,omitempty"`
}

// Tool definition for Claude requests.
type Tool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema ToolInputSchema `json:"input_schema"`
}

// Message represents a Claude conversation message.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// CacheControl represents cache directive metadata.
type CacheControl struct {
	Type string `json:"type"`
}

// RequestTextBlock is a text block with optional cache control.
type RequestTextBlock struct {
	Type         string        `json:"type"`
	Text         string        `json:"text"`
	CacheControl *CacheControl `json:"cache_control,omitempty"`
}

// MessageRequest is the Claude messages API request payload.
type MessageRequest struct {
	Model     string            `json:"model"`
	Messages  []Message         `json:"messages"`
	System    []RequestTextBlock `json:"system,omitempty"`
	MaxTokens int32             `json:"max_tokens"`
	Temperature *float32        `json:"temperature,omitempty"`
	TopP      *float32          `json:"top_p,omitempty"`
	TopK      *float32          `json:"top_k,omitempty"`
	Tools     []Tool            `json:"tools,omitempty"`
	Stream    bool              `json:"stream,omitempty"`
}

// ContentBlock represents response content blocks.
type ContentBlock struct {
	Type  json.RawMessage `json:"type"`
	Text  json.RawMessage `json:"text,omitempty"`
	ID    json.RawMessage `json:"id,omitempty"`
	Name  json.RawMessage `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`
}

// TextContentBlock is a text response block.
type TextContentBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// ToolUseContentBlock describes tool invocation payload.
type ToolUseContentBlock struct {
	Type  string          `json:"type"`
	ID    string          `json:"id"`
	Name  string          `json:"name"`
	Input json.RawMessage `json:"input"`
}

// ToolCall contains structured tool call data returned by Claude.
type ToolCall struct {
	ID    string          `json:"id"`
	Name  string          `json:"name"`
	Input json.RawMessage `json:"input"`
}

// Usage captures token accounting information.
type Usage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
}

// MessageResponse is the Claude messages API response.
type MessageResponse struct {
	ID           string         `json:"id"`
	Type         string         `json:"type"`
	Role         string         `json:"role"`
	Content      []ContentBlock `json:"content"`
	Model        string         `json:"model"`
	StopReason   string         `json:"stop_reason"`
	StopSequence string         `json:"stop_sequence"`
	Usage        Usage          `json:"usage"`
}

// New creates a new Claude provider instance.
func New(log logger.Logger, apiKey, modelName string) (*Provider, error) {
	if apiKey == "" {
		apiKey = os.Getenv("CLAUDE_API_KEY")
		if apiKey == "" {
			return nil, errors.New("CLAUDE_API_KEY not provided")
		}
	}

	if modelName == "" {
		modelName = os.Getenv("CLAUDE_MODEL")
		if modelName == "" {
			modelName = defaultClaudeModel
		}
	}

	baseURL := os.Getenv("CLAUDE_BASE_URL")
	if baseURL == "" {
		baseURL = defaultClaudeBaseURL
	}

	client := &http.Client{Timeout: defaultClaudeTimeout}

	return &Provider{
		client:    client,
		logger:    log,
		apiKey:    apiKey,
		modelName: modelName,
		baseURL:   baseURL,
	}, nil
}

// GetModelName returns the current Claude model identifier.
func (p *Provider) GetModelName() string {
	return p.modelName
}

// GenerateText performs a non-streaming Claude request.
func (p *Provider) GenerateText(ctx context.Context, prompt string, opts ...llm.GenerationOption) (string, *llm.UsageInfo, error) {
	options := &llm.GenerationOptions{
		Temperature: llm.ValuePtr(float32(0.7)),
		MaxTokens:   llm.ValuePtr(int32(4096)),
	}
	for _, opt := range opts {
		opt(options)
	}

	systemInstruction := options.System
	if options.Language != "" && options.Language != "en" {
		systemInstruction += fmt.Sprintf(" Please respond in %s language.", utils.GetLangName(options.Language))
	}

	reqPayload := MessageRequest{
		Model: p.modelName,
		Messages: []Message{
			{Role: "user", Content: prompt},
		},
		MaxTokens:   *options.MaxTokens,
		Temperature: options.Temperature,
		TopP:        options.TopP,
		TopK:        options.TopK,
	}

	var systemBlocks []RequestTextBlock
	cacheUsed := false

	if len(options.SystemBlocks) > 0 {
		for _, block := range options.SystemBlocks {
			textBlock := RequestTextBlock{Type: "text", Text: block.Text}
			if block.UseCache {
				textBlock.CacheControl = &CacheControl{Type: "ephemeral"}
				cacheUsed = true
			}
			systemBlocks = append(systemBlocks, textBlock)
		}
	}

	if systemInstruction != "" {
		systemBlocks = append(systemBlocks, RequestTextBlock{Type: "text", Text: systemInstruction})
	}

	if len(systemBlocks) > 0 {
		reqPayload.System = systemBlocks
	}

	if len(options.Tools) > 0 {
		claudeTools := make([]Tool, 0, len(options.Tools))
		for _, tool := range options.Tools {
			schemaMap, err := llm.ConvertSchemaToMap(tool.InputSchema)
			if err != nil {
				p.logger.Error(fmt.Sprintf("Failed to convert input schema for tool '%s'", tool.Name), err)
				return "", nil, fmt.Errorf("failed to convert input schema for tool '%s': %w", tool.Name, err)
			}

			props := make(map[string]map[string]interface{})
			if rawProps, ok := schemaMap["properties"].(map[string]interface{}); ok {
				for key, val := range rawProps {
					propMap, ok := val.(map[string]interface{})
					if !ok {
						return "", nil, fmt.Errorf("invalid property structure for tool '%s', property '%s'", tool.Name, key)
					}
					props[key] = propMap
				}
			}

			claudeSchema := ToolInputSchema{
				Type:        "object",
				Properties:  props,
				Description: tool.InputSchema.Description,
			}
			if required, ok := schemaMap["required"].([]interface{}); ok {
				claudeSchema.Required = convertInterfaceSliceToString(required)
			}

			claudeTools = append(claudeTools, Tool{
				Name:        tool.Name,
				Description: tool.Description,
				InputSchema: claudeSchema,
			})
		}
		reqPayload.Tools = claudeTools
	} else if options.ResponseSchema != nil {
		schemaJSON, err := llm.ConvertToJSONSchema(options.ResponseSchema)
		if err != nil {
			p.logger.Error("Failed to convert response schema for Claude structured output", err)
			return "", nil, fmt.Errorf("failed to convert response schema to JSON: %w", err)
		}
		if len(reqPayload.System) > 0 {
			idx := len(reqPayload.System) - 1
			reqPayload.System[idx].Text += fmt.Sprintf("\n\nPlease provide your response strictly in the following JSON format, enclosed within ```json ... ```:\n```json\n%s\n```", schemaJSON)
		} else {
			reqPayload.System = append(reqPayload.System, RequestTextBlock{Type: "text", Text: fmt.Sprintf("Please respond using the following JSON schema:\n```json\n%s\n```", schemaJSON)})
		}
	}

	body, err := json.Marshal(reqPayload)
	if err != nil {
		p.logger.Error("Failed to marshal Claude request payload", err)
		return "", nil, fmt.Errorf("failed to marshal request payload: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/messages", bytes.NewBuffer(body))
	if err != nil {
		p.logger.Error("Failed to create Claude HTTP request", err)
		return "", nil, fmt.Errorf("failed to create HTTP request: %w", err)
	}

	req.Header.Set("x-api-key", p.apiKey)
	req.Header.Set("anthropic-version", claudeAPIVersion)
	req.Header.Set("Content-Type", "application/json")
	if cacheUsed {
		req.Header.Set("anthropic-beta", "prompt-caching-2024-07-31")
		p.logger.Info("Claude prompt caching enabled for this request.")
	}

	resp, err := p.client.Do(req)
	if err != nil {
		p.logger.Error(fmt.Sprintf("Failed to send request to Claude API: %v", err), err)
		return "", nil, fmt.Errorf("failed to call Claude API: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		var errorBody map[string]interface{}
		_ = json.NewDecoder(resp.Body).Decode(&errorBody)
		p.logger.Error(fmt.Sprintf("Claude API returned non-OK status: %d - Body: %v", resp.StatusCode, errorBody), nil)
		return "", nil, fmt.Errorf("Claude API error: status code %d, details: %v", resp.StatusCode, errorBody)
	}

	var claudeResp MessageResponse
	if err := json.NewDecoder(resp.Body).Decode(&claudeResp); err != nil {
		p.logger.Error("Failed to decode Claude API response", err)
		return "", nil, fmt.Errorf("failed to decode API response: %w", err)
	}

	usage := &llm.UsageInfo{
		InputTokens:     claudeResp.Usage.InputTokens,
		OutputTokens:    claudeResp.Usage.OutputTokens,
		CacheMissTokens: claudeResp.Usage.CacheCreationInputTokens,
		CacheHitTokens:  claudeResp.Usage.CacheReadInputTokens,
	}

	if claudeResp.StopReason == "tool_use" {
		toolCallsJSON, err := extractToolCallsJSON(claudeResp.Content)
		if err != nil {
			p.logger.Error("Failed to marshal tool calls to JSON", err)
			return "", nil, fmt.Errorf("failed to marshal tool calls: %w", err)
		}
		return "TOOL_CALL::" + toolCallsJSON, usage, nil
	}

	var textBuilder strings.Builder
	for _, block := range claudeResp.Content {
		var blockType struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal(block.Type, &blockType.Type); err != nil {
			continue
		}
		if blockType.Type == "text" {
			var textBlock TextContentBlock
			if err := json.Unmarshal(block.Text, &textBlock.Text); err == nil {
				textBuilder.WriteString(textBlock.Text)
			}
		}
	}

	generatedText := textBuilder.String()
	if generatedText == "" {
		p.logger.Warning("No text content blocks found in Claude response")
		return "", nil, errors.New("no text content generated by Claude")
	}

	if len(options.Tools) == 0 && options.ResponseSchema != nil {
		if extracted, err := utils.ExtractJSONFromString(generatedText); err == nil {
			generatedText = extracted
		} else {
			p.logger.Warning(fmt.Sprintf("Failed to extract JSON from Claude response: %v", err))
		}
	}

	p.logger.Info(fmt.Sprintf("Generated text (Claude): %s", generatedText))
	return generatedText, usage, nil
}

// GenerateTextStream handles streaming responses from Claude.
func (p *Provider) GenerateTextStream(ctx context.Context, prompt string, outChan chan<- llm.StreamChunk, opts ...llm.GenerationOption) (*llm.UsageInfo, error) {
	defer func() {
		close(outChan)
	}()

	options := &llm.GenerationOptions{
		Temperature: llm.ValuePtr(float32(0.7)),
		MaxTokens:   llm.ValuePtr(int32(4096)),
	}
	for _, opt := range opts {
		opt(options)
	}

	systemInstruction := options.System
	if options.Language != "" && options.Language != "en" {
		langName := utils.GetLangName(options.Language)
		if langName != "" {
			systemInstruction += fmt.Sprintf(" Please respond in %s language.", langName)
		}
	}

	var systemBlocks []RequestTextBlock
	cacheUsed := false
	if len(options.SystemBlocks) > 0 {
		for _, block := range options.SystemBlocks {
			textBlock := RequestTextBlock{Type: "text", Text: block.Text}
			if block.UseCache {
				textBlock.CacheControl = &CacheControl{Type: "ephemeral"}
				cacheUsed = true
			}
			systemBlocks = append(systemBlocks, textBlock)
		}
	}
	if systemInstruction != "" {
		systemBlocks = append(systemBlocks, RequestTextBlock{Type: "text", Text: systemInstruction})
	}

	messages := []Message{{Role: "user", Content: prompt}}

	var claudeTools []Tool
	if len(options.Tools) > 0 {
		p.logger.Warning("Claude streaming with tools may produce partial tool events; parsing is limited to text deltas.")
		claudeTools = make([]Tool, 0, len(options.Tools))
		for _, tool := range options.Tools {
			schemaMap, err := llm.ConvertSchemaToMap(tool.InputSchema)
			if err != nil {
				p.logger.Error(fmt.Sprintf("failed to convert property schema for tool '%s'", tool.Name), err)
				outChan <- llm.StreamChunk{Err: fmt.Errorf("failed to convert property schema for tool '%s': %w", tool.Name, err)}
				return nil, err
			}

			props := make(map[string]map[string]interface{})
			if rawProps, ok := schemaMap["properties"].(map[string]interface{}); ok {
				for key, val := range rawProps {
					propMap, ok := val.(map[string]interface{})
					if !ok {
						return nil, fmt.Errorf("invalid property map for tool '%s', property '%s'", tool.Name, key)
					}
					props[key] = propMap
				}
			}

			claudeSchema := ToolInputSchema{
				Type:        "object",
				Properties:  props,
				Description: tool.InputSchema.Description,
			}
			if required, ok := schemaMap["required"].([]interface{}); ok {
				claudeSchema.Required = convertInterfaceSliceToString(required)
			}

			claudeTools = append(claudeTools, Tool{
				Name:        tool.Name,
				Description: tool.Description,
				InputSchema: claudeSchema,
			})
		}
	} else if options.ResponseSchema != nil {
		schemaJSON, err := llm.ConvertToJSONSchema(options.ResponseSchema)
		if err != nil {
			p.logger.Error("Failed to convert response schema to JSON for Claude stream", err)
			outChan <- llm.StreamChunk{Err: fmt.Errorf("failed to convert response schema: %w", err)}
			return nil, err
		}
		injectionText := fmt.Sprintf("Please provide your response strictly in the following JSON format, enclosed within ```json ... ```:\n```json\n%s\n```", schemaJSON)
		if len(systemBlocks) > 0 {
			idx := len(systemBlocks) - 1
			systemBlocks[idx].Text += "\n\n" + injectionText
		} else {
			systemBlocks = append(systemBlocks, RequestTextBlock{Type: "text", Text: injectionText})
		}
	}

	reqPayload := MessageRequest{
		Model:       p.modelName,
		Messages:    messages,
		System:      systemBlocks,
		MaxTokens:   *options.MaxTokens,
		Temperature: options.Temperature,
		TopP:        options.TopP,
		TopK:        options.TopK,
		Tools:       claudeTools,
		Stream:      true,
	}

	body, err := json.Marshal(reqPayload)
	if err != nil {
		p.logger.Error("Failed to marshal Claude stream request payload", err)
		outChan <- llm.StreamChunk{Err: fmt.Errorf("failed to marshal request payload: %w", err)}
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/messages", bytes.NewBuffer(body))
	if err != nil {
		p.logger.Error("Failed to create Claude stream HTTP request", err)
		outChan <- llm.StreamChunk{Err: fmt.Errorf("failed to create HTTP request: %w", err)}
		return nil, err
	}

	req.Header.Set("x-api-key", p.apiKey)
	req.Header.Set("anthropic-version", claudeAPIVersion)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	req.Header.Set("Cache-Control", "no-cache")
	req.Header.Set("Connection", "keep-alive")
	if cacheUsed {
		req.Header.Set("anthropic-beta", "prompt-caching-2024-07-31")
		p.logger.Info("Claude prompt caching enabled for this stream request.")
	}

	resp, err := p.client.Do(req)
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			p.logger.Info("Context cancelled during Claude stream HTTP request")
			return nil, err
		}
		p.logger.Error(fmt.Sprintf("Failed to send stream request to Claude API: %v", err), err)
		wrappedErr := fmt.Errorf("failed to call Claude API: %w", err)
		outChan <- llm.StreamChunk{Err: wrappedErr}
		return nil, wrappedErr
	}

	if resp.StatusCode != http.StatusOK {
		bodyBytes, readErr := io.ReadAll(resp.Body)
		resp.Body.Close()
		if readErr != nil {
			p.logger.Error(fmt.Sprintf("Claude API stream error: status %d, failed to read body", resp.StatusCode), readErr)
			err := fmt.Errorf("Claude API stream error: status code %d", resp.StatusCode)
			outChan <- llm.StreamChunk{Err: err}
			return nil, err
		}

		var errorResp struct {
			Type  string `json:"type"`
			Error struct {
				Type    string `json:"type"`
				Message string `json:"message"`
			} `json:"error"`
		}
		msg := fmt.Sprintf("Claude API stream error: status code %d, raw body: %s", resp.StatusCode, string(bodyBytes))
		if json.Unmarshal(bodyBytes, &errorResp) == nil && errorResp.Error.Message != "" {
			msg = fmt.Sprintf("Claude API stream error: status code %d, type: %s, message: %s", resp.StatusCode, errorResp.Error.Type, errorResp.Error.Message)
		}
		p.logger.Error(msg, nil)
		err := errors.New(msg)
		outChan <- llm.StreamChunk{Err: err}
		return nil, err
	}
	defer resp.Body.Close()

	reader := bufio.NewReader(resp.Body)
	usage := &llm.UsageInfo{}
	var currentEvent []byte

	for {
		select {
		case <-ctx.Done():
			p.logger.Info("Context cancelled during Claude stream processing")
			return usage, ctx.Err()
		default:
		}

		line, err := reader.ReadBytes('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			p.logger.Error("Error reading Claude stream", err)
			readErr := fmt.Errorf("stream read error: %w", err)
			outChan <- llm.StreamChunk{Err: readErr}
			return usage, readErr
		}

		trimmed := bytes.TrimSpace(line)
		if len(trimmed) == 0 {
			continue
		}

		if bytes.HasPrefix(trimmed, []byte("event: ")) {
			currentEvent = bytes.TrimSpace(bytes.TrimPrefix(trimmed, []byte("event: ")))
			continue
		}

		if !bytes.HasPrefix(trimmed, []byte("data: ")) {
			continue
		}

		data := bytes.TrimPrefix(trimmed, []byte("data: "))
		if len(data) == 0 {
			continue
		}

		var streamEvent StreamEvent
		if err := json.Unmarshal(data, &streamEvent); err != nil {
			p.logger.Error(fmt.Sprintf("Failed to unmarshal Claude stream data: %v", err), err)
			continue
		}

		if currentEvent != nil && streamEvent.Type != string(currentEvent) {
			p.logger.Warning(fmt.Sprintf("Mismatched event type: header=%s payload=%s", string(currentEvent), streamEvent.Type))
		}

		switch streamEvent.Type {
		case "message_start":
			if streamEvent.Message != nil {
				usage.InputTokens = streamEvent.Message.Usage.InputTokens
				usage.OutputTokens = streamEvent.Message.Usage.OutputTokens
				usage.CacheCreateTokens = streamEvent.Message.Usage.CacheCreationInputTokens
				usage.CacheHitTokens = streamEvent.Message.Usage.CacheReadInputTokens
			}
		case "content_block_delta":
			if streamEvent.Delta != nil && streamEvent.Delta.Type == "text_delta" {
				outChan <- llm.StreamChunk{Delta: streamEvent.Delta.Text}
			}
		case "message_delta":
			if streamEvent.Usage != nil {
				usage.InputTokens = streamEvent.Usage.InputTokens
				usage.OutputTokens = streamEvent.Usage.OutputTokens
				usage.CacheCreateTokens = streamEvent.Usage.CacheCreationInputTokens
				usage.CacheHitTokens = streamEvent.Usage.CacheReadInputTokens
			}
		case "message_stop":
			outChan <- llm.StreamChunk{IsFinal: true}
		}
	}

	return usage, nil
}

// Close releases resources.
func (p *Provider) Close() error {
	p.logger.Info("[Claude] Provider closed.")
	return nil
}

func convertInterfaceSliceToString(values []interface{}) []string {
	result := make([]string, 0, len(values))
	for _, v := range values {
		if str, ok := v.(string); ok {
			result = append(result, str)
		}
	}
	return result
}

func extractToolCallsJSON(blocks []ContentBlock) (string, error) {
	var toolCalls []ToolCall
	for _, block := range blocks {
		var blockType struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal(block.Type, &blockType.Type); err != nil {
			continue
		}
		if blockType.Type != "tool_use" {
			continue
		}

		var toolBlock ToolUseContentBlock
		if err := json.Unmarshal(block.ID, &toolBlock.ID); err != nil {
			continue
		}
		if err := json.Unmarshal(block.Name, &toolBlock.Name); err != nil {
			continue
		}
		toolBlock.Input = block.Input
		toolCalls = append(toolCalls, ToolCall{ID: toolBlock.ID, Name: toolBlock.Name, Input: toolBlock.Input})
	}

	if len(toolCalls) == 0 {
		return "", errors.New("no tool calls found in Claude response")
	}

	bytes, err := json.Marshal(toolCalls)
	if err != nil {
		return "", err
	}
	return string(bytes), nil
}
