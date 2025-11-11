package cerebras

import (
	"context"
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
	defaultModel   = "qwen-3-235b-a22b"
	defaultBaseURL = "https://api.cerebras.ai/v1"
)

// Provider implements llm.Provider for Cerebras models.
type Provider struct {
	client    sdk.Client
	logger    logger.Logger
	modelName string
}

// New creates a new Cerebras provider instance.
func New(log logger.Logger, apiKey, modelName string) (*Provider, error) {
	if apiKey == "" {
		apiKey = os.Getenv("CEREBRAS_API_KEY")
		if apiKey == "" {
			return nil, errors.New("CEREBRAS_API_KEY not provided")
		}
	}

	if modelName == "" {
		modelName = os.Getenv("CEREBRAS_MODEL")
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
func (p *Provider) GetModelName() string { return p.modelName }

// GenerateText performs a non-streaming Cerebras request.
func (p *Provider) GenerateText(ctx context.Context, prompt string, opts ...llm.GenerationOption) (string, *llm.UsageInfo, error) {
	options := &llm.GenerationOptions{
		Temperature: llm.ValuePtr(float32(0.6)),
		MaxTokens:   llm.ValuePtr(int32(40000)),
		TopP:        llm.ValuePtr(float32(0.95)),
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
		p.logger.Error("[Cerebras] Failed to generate content", err)
		return "", nil, err
	}

	if len(resp.Choices) == 0 || resp.Choices[0].Message.Content == "" {
		p.logger.Warning("[Cerebras] No content generated")
		return "", nil, errors.New("no content generated")
	}

	generated := resp.Choices[0].Message.Content
	if options.ResponseSchema != nil {
		if extracted, extractErr := utils.ExtractJSONFromString(generated); extractErr == nil {
			generated = extracted
		} else {
			p.logger.Warningf("[Cerebras] Failed to extract JSON: %v", extractErr)
		}
	}

	usage := &llm.UsageInfo{
		InputTokens:  int(resp.Usage.PromptTokens),
		OutputTokens: int(resp.Usage.CompletionTokens),
	}

	p.logger.Info(fmt.Sprintf("Generated text (Cerebras): %s", generated))
	return generated, usage, nil
}

// GenerateTextStream streams responses from Cerebras.
func (p *Provider) GenerateTextStream(ctx context.Context, prompt string, outChan chan<- llm.StreamChunk, opts ...llm.GenerationOption) (*llm.UsageInfo, error) {
	defer close(outChan)

	options := &llm.GenerationOptions{
		Temperature: llm.ValuePtr(float32(0.6)),
		MaxTokens:   llm.ValuePtr(int32(40000)),
		TopP:        llm.ValuePtr(float32(0.95)),
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

	var usage *llm.UsageInfo
	var full strings.Builder

	for stream.Next() {
		resp := stream.Current()

		if len(resp.Choices) > 0 {
			delta := resp.Choices[0].Delta.Content
			if delta != "" {
				full.WriteString(delta)
				select {
				case outChan <- llm.StreamChunk{Delta: delta}:
				case <-ctx.Done():
					p.logger.Info("[Cerebras Stream] Context cancelled during send")
					return usage, ctx.Err()
				}
			}
		}

		if resp.Usage.PromptTokens > 0 || resp.Usage.CompletionTokens > 0 {
			usage = &llm.UsageInfo{
				InputTokens:  int(resp.Usage.PromptTokens),
				OutputTokens: int(resp.Usage.CompletionTokens),
			}
		}
	}

	if err := stream.Err(); err != nil {
		p.logger.Error("[Cerebras Stream] Stream error", err)
		return usage, err
	}

	if options.ResponseSchema != nil {
		if extracted, extractErr := utils.ExtractJSONFromString(full.String()); extractErr == nil {
			p.logger.Info(fmt.Sprintf("[Cerebras Stream] Extracted JSON: %s", extracted))
		} else {
			p.logger.Warningf("[Cerebras Stream] Failed to extract JSON: %v", extractErr)
		}
	}

	return usage, nil
}

// Close releases resources.
func (p *Provider) Close() error {
	p.logger.Info("[Cerebras] Provider closed.")
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
		if err == nil {
			builder.WriteString("Please provide your response strictly in the following JSON format, enclosed within ```json ... ```:\n```json\n")
			builder.WriteString(schemaJSON)
			builder.WriteString("\n```\n\n")
		}
	}
	if options.ResponseFormat != "" {
		builder.WriteString(fmt.Sprintf("Response format: %s\n\n", options.ResponseFormat))
	}

	return strings.TrimSpace(builder.String())
}
