package inception

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
	defaultModel               = "mercury"
	baseURL                    = "https://api.inceptionlabs.ai/v1"
	structuredOutputSchemaName = "structured_output"
)

// Provider implements llm.Provider for Inception models.
type Provider struct {
	client    sdk.Client
	apiKey    string
	modelName string
	logger    logger.Logger
}

// New creates a new Inception provider.
func New(log logger.Logger, apiKey, modelName string) (*Provider, error) {
	resolvedKey := apiKey
	if resolvedKey == "" {
		resolvedKey = os.Getenv("INCEPTION_API_KEY")
		if resolvedKey == "" {
			log.Info("[Inception] API key not explicitly provided, relying on INCEPTION_API_KEY environment variable.")
		}
	}

	if modelName == "" {
		modelName = defaultModel
	}

	client := sdk.NewClient(
		option.WithAPIKey(resolvedKey),
		option.WithBaseURL(baseURL),
	)

	return &Provider{client: client, apiKey: resolvedKey, modelName: modelName, logger: log}, nil
}

// GetModelName returns the active model name.
func (p *Provider) GetModelName() string {
	return p.modelName
}

// GenerateText issues a non-streaming completion request.
func (p *Provider) GenerateText(ctx context.Context, prompt string, opts ...llm.GenerationOption) (string, *llm.UsageInfo, error) {
	options := &llm.GenerationOptions{
		Temperature: llm.ValuePtr(float32(0.7)),
		MaxTokens:   llm.ValuePtr(int32(2048)),
	}
	for _, opt := range opts {
		opt(options)
	}

	systemPrompt := composeSystemPrompt(options)

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
		if err := p.applyTools(&params, options.Tools); err != nil {
			return "", nil, err
		}
	} else if options.ResponseSchema != nil {
		p.applyStructuredOutput(&params, options.ResponseSchema)
	}

	resp, err := p.client.Chat.Completions.New(ctx, params)
	if err != nil {
		p.logger.Error("[Inception] API error", err)
		return "", nil, err
	}

	if len(resp.Choices) == 0 {
		p.logger.Warning("[Inception] No choices returned")
		return "", nil, nil
	}

	choice := resp.Choices[0]
	usage := &llm.UsageInfo{
		InputTokens:  int(resp.Usage.PromptTokens),
		OutputTokens: int(resp.Usage.CompletionTokens),
	}

	if len(choice.Message.ToolCalls) > 0 {
		p.logger.Infof("[Inception] Received %d tool call(s)", len(choice.Message.ToolCalls))
		return choice.Message.ToolCalls[0].Function.Arguments, usage, nil
	}

	if resp.SystemFingerprint != "" {
		p.logger.Info("[Inception] System Fingerprint: " + resp.SystemFingerprint)
	}

	return choice.Message.Content, usage, nil
}

// GenerateTextStream streams completion chunks.
func (p *Provider) GenerateTextStream(ctx context.Context, prompt string, outChan chan<- llm.StreamChunk, opts ...llm.GenerationOption) (*llm.UsageInfo, error) {
	defer close(outChan)

	options := &llm.GenerationOptions{
		Temperature: llm.ValuePtr(float32(0.7)),
		MaxTokens:   llm.ValuePtr(int32(1024)),
	}
	for _, opt := range opts {
		opt(options)
	}

	systemPrompt := composeSystemPrompt(options)

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
		p.logger.Warning("[Inception Stream] Tool calling is disabled for streaming; ignoring tools.")
	} else if options.ResponseSchema != nil {
		p.logger.Warning("[Inception Stream] Structured output is not supported in streaming mode; ignoring schema.")
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
					p.logger.Info("[Inception Stream] Context cancelled during send")
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
		p.logger.Error("[Inception Stream] Stream error", err)
		usage, _ := processFinalUsage(lastUsage, p.logger)
		return usage, err
	}

	usage, err := processFinalUsage(lastUsage, p.logger)
	if err != nil {
		p.logger.Errorf("[Inception Stream] Usage processing error: %v", err)
	}
	if systemFingerprint != "" {
		p.logger.Info("[Inception Stream] System Fingerprint: " + systemFingerprint)
	}

	return usage, nil
}

// Close closes any persistent resources (none for now).
func (p *Provider) Close() error {
	p.logger.Info("[Inception] Provider closed.")
	return nil
}

func (p *Provider) applyTools(params *sdk.ChatCompletionNewParams, tools []*llm.Tool) error {
	params.Tools = make([]sdk.ChatCompletionToolParam, 0, len(tools))
	for _, tool := range tools {
		if tool.InputSchema == nil {
			p.logger.Warningf("[Inception] Tool '%s' missing input schema, skipping", tool.Name)
			continue
		}

		schemaMap, err := llm.ConvertSchemaToMap(tool.InputSchema)
		if err != nil {
			p.logger.Errorf("[Inception] Failed to convert schema for tool '%s': %v", tool.Name, err)
			return err
		}

		params.Tools = append(params.Tools, sdk.ChatCompletionToolParam{
			Function: sdk.FunctionDefinitionParam{
				Name:        tool.Name,
				Description: sdk.String(tool.Description),
				Parameters:  schemaMap,
			},
		})
	}

	p.logger.Info("[Inception] Using tool calling mode")
	return nil
}

func (p *Provider) applyStructuredOutput(params *sdk.ChatCompletionNewParams, schema *llm.SchemaProperty) {
	schemaMap, err := llm.ConvertSchemaToMap(schema)
	if err != nil {
		p.logger.Error("[Inception] Failed to convert response schema", err)
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

	p.logger.Info("[Inception] Using structured output mode")
}

func composeSystemPrompt(options *llm.GenerationOptions) string {
	systemPrompt := options.System
	if options.Language != "" {
		if options.Language == "ko" || options.Language == "korean" {
			systemPrompt = "해당 언어로 작성하라. " + systemPrompt
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
				systemPrompt = "해당 언어로 작성하라. " + sb.String()
			} else {
				systemPrompt = "Please write in " + options.Language + ". " + sb.String()
			}
		} else {
			systemPrompt = sb.String()
		}
	}
	return systemPrompt
}

func processFinalUsage(lastUsage *sdk.CompletionUsage, log logger.Logger) (*llm.UsageInfo, error) {
	if lastUsage == nil {
		log.Warning("[Inception Stream] No usage information received")
		return nil, nil
	}

	usage := &llm.UsageInfo{
		InputTokens:  int(lastUsage.PromptTokens),
		OutputTokens: int(lastUsage.CompletionTokens),
	}
	return usage, nil
}
