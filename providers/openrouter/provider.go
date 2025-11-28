package openrouter

import (
	"context"
	"errors"
	"io"
	"os"
	"strings"

	sdk "github.com/openai/openai-go"
	"github.com/openai/openai-go/option"

	"github.com/ulgerang/llm-module/llm"
	"github.com/ulgerang/llm-module/logger"
)

const (
	apiBaseURL                  = "https://openrouter.ai/api/v1"
	defaultModel                = "openai/gpt-4-turbo-preview"
	structuredOutputSchemaName  = "structured_output"
)

// Provider implements llm.Provider for OpenRouter using the OpenAI-compatible API.
type Provider struct {
	client    sdk.Client
	apiKey    string
	modelName string
	logger    logger.Logger
}

// New creates a new OpenRouter provider using the OpenAI Go SDK.
func New(log logger.Logger, apiKey, modelName string) (*Provider, error) {
	resolvedKey := apiKey
	if resolvedKey == "" {
		resolvedKey = os.Getenv("OPENROUTER_API_KEY")
		if resolvedKey == "" {
			return nil, errors.New("OPENROUTER_API_KEY not provided")
		}
	}

	if modelName == "" {
		modelName = defaultModel
	}

	client := sdk.NewClient(
		option.WithAPIKey(resolvedKey),
		option.WithBaseURL(apiBaseURL),
		option.WithHeader("HTTP-Referer", "https://chatsite.ai"),
		option.WithHeader("X-Title", "ChatSite AI"),
	)

	return &Provider{client: client, apiKey: resolvedKey, modelName: modelName, logger: log}, nil
}

// GetModelName returns the configured OpenRouter model name.
func (p *Provider) GetModelName() string {
	return p.modelName
}

// GenerateText performs a non-streaming OpenRouter request.
func (p *Provider) GenerateText(ctx context.Context, prompt string, opts ...llm.GenerationOption) (string, *llm.UsageInfo, error) {
	options := &llm.GenerationOptions{
		Temperature: llm.ValuePtr(float32(0.7)),
		MaxTokens:   llm.ValuePtr(int32(2048)),
	}
	for _, opt := range opts {
		opt(options)
	}

	systemPrompt := p.composeSystemPrompt(options)

	messages := []sdk.ChatCompletionMessageParamUnion{}
	if systemPrompt != "" {
		messages = append(messages, sdk.SystemMessage(systemPrompt))
	}
	messages = append(messages, sdk.UserMessage(prompt))

	params := sdk.ChatCompletionNewParams{Messages: messages, Model: p.modelName}

	if options.Temperature != nil {
		params.Temperature = sdk.Float(float64(*options.Temperature))
	}
	if options.MaxTokens != nil {
		params.MaxTokens = sdk.Int(int64(*options.MaxTokens))
	}
	if options.TopP != nil {
		params.TopP = sdk.Float(float64(*options.TopP))
	}

	if len(options.Tools) > 0 {
		p.applyTools(&params, options)
	} else if options.ResponseSchema != nil {
		p.applyStructuredOutput(&params, options)
	}

	resp, err := p.client.Chat.Completions.New(ctx, params)
	if err != nil {
		p.logger.Error("[OpenRouter] API error", err)
		return "", nil, err
	}

	if len(resp.Choices) == 0 {
		p.logger.Warning("[OpenRouter] No choices returned")
		return "", nil, nil
	}

	choice := resp.Choices[0]
	usage := &llm.UsageInfo{
		InputTokens:  int(resp.Usage.PromptTokens),
		OutputTokens: int(resp.Usage.CompletionTokens),
	}

	if len(choice.Message.ToolCalls) > 0 {
		p.logger.Infof("[OpenRouter] Received %d tool call(s)", len(choice.Message.ToolCalls))
		return choice.Message.ToolCalls[0].Function.Arguments, usage, nil
	}

	if resp.SystemFingerprint != "" {
		p.logger.Info("[OpenRouter] System Fingerprint: " + resp.SystemFingerprint)
	}

	return choice.Message.Content, usage, nil
}

// GenerateTextStream streams responses from OpenRouter.
func (p *Provider) GenerateTextStream(ctx context.Context, prompt string, outChan chan<- llm.StreamChunk, opts ...llm.GenerationOption) (*llm.UsageInfo, error) {
	defer close(outChan)

	options := &llm.GenerationOptions{
		Temperature: llm.ValuePtr(float32(0.7)),
		MaxTokens:   llm.ValuePtr(int32(1024)),
	}
	for _, opt := range opts {
		opt(options)
	}

	systemPrompt := p.composeSystemPrompt(options)

	messages := []sdk.ChatCompletionMessageParamUnion{}
	if systemPrompt != "" {
		messages = append(messages, sdk.SystemMessage(systemPrompt))
	}
	messages = append(messages, sdk.UserMessage(prompt))

	params := sdk.ChatCompletionNewParams{Messages: messages, Model: p.modelName}

	if options.Temperature != nil {
		params.Temperature = sdk.Float(float64(*options.Temperature))
	}
	if options.MaxTokens != nil {
		params.MaxTokens = sdk.Int(int64(*options.MaxTokens))
	}
	if options.TopP != nil {
		params.TopP = sdk.Float(float64(*options.TopP))
	}

	if len(options.Tools) > 0 {
		p.logger.Warning("[OpenRouter Stream] Tool calling not supported, ignoring tools.")
	} else if options.ResponseSchema != nil {
		p.logger.Warning("[OpenRouter Stream] Structured output not supported for streaming, ignoring schema.")
	}

	stream := p.client.Chat.Completions.NewStreaming(ctx, params)
	defer stream.Close()

	var lastUsage *sdk.CompletionUsage
	var systemFingerprint string

	for stream.Next() {
		resp := stream.Current()
		if len(resp.Choices) > 0 {
			delta := resp.Choices[0].Delta.Content
			if delta != "" {
				select {
				case outChan <- llm.StreamChunk{Delta: delta}:
				case <-ctx.Done():
					p.logger.Info("[OpenRouter Stream] Context cancelled during send")
					usage, _ := processFinalUsage(lastUsage, p.logger)
					return usage, ctx.Err()
				}
			}
		}

		lastUsage = &resp.Usage
		if resp.SystemFingerprint != "" {
			systemFingerprint = resp.SystemFingerprint
		}
	}

	if err := stream.Err(); err != nil && !errors.Is(err, io.EOF) {
		p.logger.Error("[OpenRouter Stream] Stream error", err)
		usage, _ := processFinalUsage(lastUsage, p.logger)
		return usage, err
	}

	usage, err := processFinalUsage(lastUsage, p.logger)
	if err != nil {
		p.logger.Errorf("[OpenRouter Stream] Usage processing error: %v", err)
	}
	if systemFingerprint != "" {
		p.logger.Info("[OpenRouter Stream] System Fingerprint: " + systemFingerprint)
	}

	return usage, nil
}

// Close releases resources.
func (p *Provider) Close() error {
	p.logger.Info("[OpenRouter] Provider closed.")
	return nil
}

func (p *Provider) composeSystemPrompt(options *llm.GenerationOptions) string {
	systemPrompt := options.System
	if options.Language != "" {
		if options.Language == "ko" || options.Language == "korean" {
			systemPrompt = "?대떦 ?몄뼱濡??묒꽦?섎씪. " + systemPrompt
		} else {
			systemPrompt = "Please write in " + options.Language + ". " + systemPrompt
		}
	}
	if len(options.SystemBlocks) > 0 {
		var sb strings.Builder
		for _, block := range options.SystemBlocks {
			sb.WriteString(block.Text)
		}
		if options.Language != "" {
			if options.Language == "ko" || options.Language == "korean" {
				systemPrompt = "?대떦 ?몄뼱濡??묒꽦?섎씪. " + sb.String()
			} else {
				systemPrompt = "Please write in " + options.Language + ". " + sb.String()
			}
		} else {
			systemPrompt = sb.String()
		}
	}
	return systemPrompt
}

func (p *Provider) applyTools(params *sdk.ChatCompletionNewParams, options *llm.GenerationOptions) {
	supported := []string{
		"openai/gpt-4-turbo-preview",
		"openai/gpt-4-turbo",
		"openai/gpt-4",
		"openai/gpt-3.5-turbo",
		"anthropic/claude-3-opus",
		"anthropic/claude-3-sonnet",
		"anthropic/claude-3-haiku",
	}

	supportsTools := false
	for _, model := range supported {
		if strings.Contains(p.modelName, model) {
			supportsTools = true
			break
		}
	}

	if !supportsTools {
		p.logger.Warningf("[OpenRouter] Model '%s' does not support tool calling. Ignoring tools.", p.modelName)
		return
	}

	params.Tools = make([]sdk.ChatCompletionToolParam, 0, len(options.Tools))
	for _, tool := range options.Tools {
		if tool.InputSchema == nil {
			p.logger.Warningf("[OpenRouter] Tool '%s' missing schema, skipping", tool.Name)
			continue
		}

		schemaMap, err := llm.ConvertSchemaToMap(tool.InputSchema)
		if err != nil {
			p.logger.Errorf("[OpenRouter] Failed to convert schema for tool '%s': %v", tool.Name, err)
			continue
		}

		params.Tools = append(params.Tools, sdk.ChatCompletionToolParam{
			Function: sdk.FunctionDefinitionParam{
				Name:        tool.Name,
				Description: sdk.String(tool.Description),
				Parameters:  schemaMap,
			},
		})
	}

	p.logger.Info("[OpenRouter] Using tool calling mode")
}

func (p *Provider) applyStructuredOutput(params *sdk.ChatCompletionNewParams, options *llm.GenerationOptions) {
	schemaMap, err := llm.ConvertSchemaToMap(options.ResponseSchema)
	if err != nil {
		p.logger.Error("[OpenRouter] Failed to convert response schema", err)
		return
	}

	schemaParam := sdk.ResponseFormatJSONSchemaJSONSchemaParam{
		Name:        structuredOutputSchemaName,
		Description: sdk.String("Structured output based on the requested schema"),
		Schema:      schemaMap,
		Strict:      sdk.Bool(true),
	}
	params.ResponseFormat = sdk.ChatCompletionNewParamsResponseFormatUnion{
		OfJSONSchema: &sdk.ResponseFormatJSONSchemaParam{JSONSchema: schemaParam},
	}

	p.logger.Info("[OpenRouter] Using structured output mode")
}

func processFinalUsage(lastUsage *sdk.CompletionUsage, log logger.Logger) (*llm.UsageInfo, error) {
	if lastUsage == nil {
		log.Warning("[OpenRouter Stream] No usage information received")
		return nil, nil
	}

	usage := &llm.UsageInfo{
		InputTokens:  int(lastUsage.PromptTokens),
		OutputTokens: int(lastUsage.CompletionTokens),
	}
	return usage, nil
}
