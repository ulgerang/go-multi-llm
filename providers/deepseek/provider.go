package deepseek

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"

	sdk "github.com/openai/openai-go"
	"github.com/openai/openai-go/option"

	"github.com/ulgerang/llm-module/llm"
	"github.com/ulgerang/llm-module/logger"
	"github.com/ulgerang/llm-module/utils"
)

const (
	defaultModel   = "deepseek-chat"
	defaultBaseURL = "https://api.deepseek.com/v1"
)

// Provider implements llm.Provider for DeepSeek using the OpenAI-compatible SDK.
type Provider struct {
	client    sdk.Client
	logger    logger.Logger
	modelName string
}

// New creates a new DeepSeek provider instance.
func New(log logger.Logger, apiKey, modelName string) (*Provider, error) {
	if apiKey == "" {
		apiKey = os.Getenv("DEEPSEEK_API_KEY")
		if apiKey == "" {
			return nil, errors.New("DEEPSEEK_API_KEY not provided")
		}
	}

	if modelName == "" {
		modelName = os.Getenv("DEEPSEEK_MODEL")
		if modelName == "" {
			modelName = defaultModel
		}
	}

	client := sdk.NewClient(
		option.WithAPIKey(apiKey),
		option.WithBaseURL(defaultBaseURL),
	)

	return &Provider{client: client, logger: log, modelName: modelName}, nil
}

// GetModelName returns the configured model name.
func (p *Provider) GetModelName() string {
	return p.modelName
}

// GenerateText performs a non-streaming DeepSeek request.
func (p *Provider) GenerateText(ctx context.Context, prompt string, opts ...llm.GenerationOption) (string, *llm.UsageInfo, error) {
	options := &llm.GenerationOptions{
		Temperature: llm.ValuePtr(float32(0.7)),
		MaxTokens:   llm.ValuePtr(int32(4096)),
		System:      "You are a helpful assistant.",
	}
	for _, opt := range opts {
		opt(options)
	}

	systemPrompt := buildSystemPrompt(options)

	messages := []sdk.ChatCompletionMessageParamUnion{}
	if systemPrompt != "" {
		messages = append(messages, sdk.SystemMessage(systemPrompt))
	}
	messages = append(messages, sdk.UserMessage(prompt))

	req := sdk.ChatCompletionNewParams{Model: p.modelName, Messages: messages}

	if options.Temperature != nil {
		req.Temperature = sdk.Float(float64(*options.Temperature))
	}
	if options.MaxTokens != nil {
		req.MaxTokens = sdk.Int(int64(*options.MaxTokens))
	}
	if options.TopP != nil {
		req.TopP = sdk.Float(float64(*options.TopP))
	}

	resp, err := p.client.Chat.Completions.New(ctx, req)
	if err != nil {
		p.logger.Error("[DeepSeek] Failed to generate content", err)
		return "", nil, err
	}

	if len(resp.Choices) == 0 || resp.Choices[0].Message.Content == "" {
		p.logger.Warning("[DeepSeek] No content generated")
		return "", nil, errors.New("no content generated")
	}

	generated := resp.Choices[0].Message.Content
	if options.ResponseSchema != nil {
		if extracted, extractErr := utils.ExtractJSONFromString(generated); extractErr == nil {
			generated = extracted
		} else {
			p.logger.Warningf("[DeepSeek] Failed to extract JSON: %v", extractErr)
		}
	}

	usage := &llm.UsageInfo{
		InputTokens:  int(resp.Usage.PromptTokens),
		OutputTokens: int(resp.Usage.CompletionTokens),
	}

	p.logger.Info(fmt.Sprintf("Generated text (DeepSeek): %s", generated))
	return generated, usage, nil
}

// GenerateTextStream streams responses from DeepSeek.
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

	systemPrompt := buildSystemPrompt(options)

	messages := []sdk.ChatCompletionMessageParamUnion{}
	if systemPrompt != "" {
		messages = append(messages, sdk.SystemMessage(systemPrompt))
	}
	messages = append(messages, sdk.UserMessage(prompt))

	req := sdk.ChatCompletionNewParams{Model: p.modelName, Messages: messages}

	if options.Temperature != nil {
		req.Temperature = sdk.Float(float64(*options.Temperature))
	}
	if options.MaxTokens != nil {
		req.MaxTokens = sdk.Int(int64(*options.MaxTokens))
	}
	if options.TopP != nil {
		req.TopP = sdk.Float(float64(*options.TopP))
	}

	stream := p.client.Chat.Completions.NewStreaming(ctx, req)
	defer stream.Close()

	var lastChunk sdk.ChatCompletionChunk

	for stream.Next() {
		chunk := stream.Current()
		lastChunk = chunk

		if len(chunk.Choices) > 0 {
			delta := chunk.Choices[0].Delta.Content
			if delta != "" {
				outChan <- llm.StreamChunk{Delta: delta}
			}
		}
	}

	if err := stream.Err(); err != nil {
		p.logger.Error("[DeepSeek] Stream error", err)
		outChan <- llm.StreamChunk{Err: err}
		return nil, err
	}

	outChan <- llm.StreamChunk{IsFinal: true}

	usage := parseUsageFromChunk(lastChunk, p.logger)
	return usage, nil
}

// Close releases resources.
func (p *Provider) Close() error {
	p.logger.Info("[DeepSeek] Provider closed.")
	return nil
}

func buildSystemPrompt(options *llm.GenerationOptions) string {
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
	if options.Language != "" && options.Language != "en" {
		builder.WriteString(fmt.Sprintf("Please respond in %s language.", utils.GetLangName(options.Language)))
		builder.WriteString("\n\n")
	}
	if options.ResponseSchema != nil {
		schemaJSON, err := llm.ConvertToJSONSchema(options.ResponseSchema)
		if err != nil {
			return strings.TrimSpace(builder.String())
		}
		builder.WriteString("Please provide your response strictly in the following JSON format, enclosed within ```json ... ```:\n```json\n")
		builder.WriteString(schemaJSON)
		builder.WriteString("\n```\n\n")
	}
	if options.ResponseFormat != "" {
		builder.WriteString(fmt.Sprintf("Response format: %s\n\n", options.ResponseFormat))
	}

	return strings.TrimSpace(builder.String())
}

func parseUsageFromChunk(chunk sdk.ChatCompletionChunk, log logger.Logger) *llm.UsageInfo {
	if chunk.RawJSON() == "" {
		return nil
	}

	type deepSeekUsage struct {
		Usage struct {
			PromptTokens        int `json:"prompt_tokens"`
			CompletionTokens    int `json:"completion_tokens"`
			PromptCacheHitTokens  int `json:"prompt_cache_hit_tokens"`
			PromptCacheMissTokens int `json:"prompt_cache_miss_tokens"`
		} `json:"usage"`
	}

	var payload deepSeekUsage
	if err := json.Unmarshal([]byte(chunk.RawJSON()), &payload); err != nil {
		log.Error("[DeepSeek] Failed to parse usage data", err)
		return nil
	}

	return &llm.UsageInfo{
		InputTokens:     payload.Usage.PromptTokens,
		OutputTokens:    payload.Usage.CompletionTokens,
		CacheHitTokens:  payload.Usage.PromptCacheHitTokens,
		CacheMissTokens: payload.Usage.PromptCacheMissTokens,
	}
}
