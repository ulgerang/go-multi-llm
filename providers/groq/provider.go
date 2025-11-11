package groq

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
	defaultModel   = "mistral-saba-24b"
	defaultBaseURL = "https://api.groq.com/openai/v1"
)

// Provider implements llm.Provider for Groq models.
type Provider struct {
	client    sdk.Client
	logger    logger.Logger
	modelName string
}

// New creates a new Groq provider.
func New(log logger.Logger, apiKey, modelName string) (*Provider, error) {
	if apiKey == "" {
		apiKey = os.Getenv("GROQ_API_KEY")
		if apiKey == "" {
			return nil, errors.New("GROQ_API_KEY not provided")
		}
	}

	if modelName == "" {
		modelName = os.Getenv("GROQ_MODEL")
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

// GetModelName returns the active Groq model name.
func (p *Provider) GetModelName() string {
	return p.modelName
}

// GenerateText performs a non-streaming Groq request.
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
	if options.Language != "" {
		messages = append(messages, sdk.UserMessage(languageReminder(options.Language)))
	}

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
		p.logger.Error("[Groq] Failed to generate content", err)
		return "", nil, err
	}

	if len(resp.Choices) == 0 || resp.Choices[0].Message.Content == "" {
		p.logger.Warning("[Groq] No content generated")
		return "", nil, errors.New("no content generated")
	}

	generated := resp.Choices[0].Message.Content
	if options.ResponseSchema != nil {
		if extracted, extractErr := utils.ExtractJSONFromString(generated); extractErr == nil {
			generated = extracted
		} else {
			p.logger.Warningf("[Groq] Failed to extract JSON: %v", extractErr)
		}
	}

	usage := &llm.UsageInfo{
		InputTokens:  int(resp.Usage.PromptTokens),
		OutputTokens: int(resp.Usage.CompletionTokens),
	}

	p.logger.Info(fmt.Sprintf("Generated text (Groq): %s", generated))
	return generated, usage, nil
}

// GenerateTextStream streams responses from Groq.
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
		p.logger.Error("[Groq] Stream error", err)
		outChan <- llm.StreamChunk{Err: err}
		return nil, err
	}

	return parseUsageFromChunk(lastChunk, p.logger), nil
}

// Close releases resources.
func (p *Provider) Close() error {
	p.logger.Info("[Groq] Provider closed.")
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
		builder.WriteString(languageReminder(options.Language))
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

	return strings.TrimSpace(builder.String())
}

func languageReminder(code string) string {
	return fmt.Sprintf("[Important!!]Please respond in **%s**.", utils.GetLangName(code))
}

func parseUsageFromChunk(chunk sdk.ChatCompletionChunk, log logger.Logger) *llm.UsageInfo {
	if chunk.RawJSON() == "" {
		return nil
	}

	type usageEnvelope struct {
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}

	var payload usageEnvelope
	if err := json.Unmarshal([]byte(chunk.RawJSON()), &payload); err != nil {
		log.Error("[Groq] Failed to parse usage data", err)
		return nil
	}

	return &llm.UsageInfo{
		InputTokens:  payload.Usage.PromptTokens,
		OutputTokens: payload.Usage.CompletionTokens,
	}
}
