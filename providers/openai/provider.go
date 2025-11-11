package openai

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
	defaultOpenAIModel         = sdk.ChatModelGPT4o
	structuredOutputSchemaName = "structured_output"
)

// Provider implements llm.Provider for OpenAI GPT models using the official SDK.
type Provider struct {
	client    sdk.Client
	apiKey    string
	modelName string
	logger    logger.Logger
}

// New creates a new Provider instance using the official Go client.
func New(log logger.Logger, apiKey, modelName string) (*Provider, error) {
	resolvedAPIKey := apiKey
	if resolvedAPIKey == "" {
		resolvedAPIKey = os.Getenv("OPENAI_API_KEY")
		if resolvedAPIKey == "" {
			log.Info("[OpenAI] API key not explicitly provided, relying on OPENAI_API_KEY environment variable.")
		}
	}

	if modelName == "" {
		modelName = defaultOpenAIModel
	}

	var client sdk.Client
	if resolvedAPIKey != "" {
		client = sdk.NewClient(option.WithAPIKey(resolvedAPIKey))
	} else {
		client = sdk.NewClient()
	}

	return &Provider{
		client:    client,
		apiKey:    resolvedAPIKey,
		modelName: modelName,
		logger:    log,
	}, nil
}

// GetModelName returns the model name used by this provider.
func (p *Provider) GetModelName() string {
	return p.modelName
}

// GenerateText generates a complete response, supporting text, JSON mode, structured output, and tool calls.
func (p *Provider) GenerateText(ctx context.Context, prompt string, opts ...llm.GenerationOption) (string, *llm.UsageInfo, error) {
	options := &llm.GenerationOptions{
		Temperature: llm.ValuePtr(float32(0.7)),
		MaxTokens:   llm.ValuePtr(int32(2048)),
	}
	for _, opt := range opts {
		opt(options)
	}

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

	messages := []sdk.ChatCompletionMessageParamUnion{}
	if systemPrompt != "" {
		messages = append(messages, sdk.SystemMessage(systemPrompt))
	}
	messages = append(messages, sdk.UserMessage(prompt))

	params := sdk.ChatCompletionNewParams{
		Messages: messages,
		Model:    p.modelName,
	}

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
		p.logger.Info("[OpenAI] Using Tool Calling mode.")
		params.Tools = make([]sdk.ChatCompletionToolParam, 0, len(options.Tools))
		for _, t := range options.Tools {
			if t.InputSchema == nil {
				p.logger.Warningf("[OpenAI] Tool '%s' has no InputSchema, skipping parameter definition.", t.Name)
				continue
			}
			schemaMap, err := llm.ConvertSchemaToMap(t.InputSchema)
			if err != nil {
				p.logger.Errorf("[OpenAI] Failed to convert schema for tool '%s': %v", t.Name, err)
				return "", nil, errors.New("failed to process tool schema for tool: " + t.Name)
			}

			toolParam := sdk.ChatCompletionToolParam{
				Function: sdk.FunctionDefinitionParam{
					Name:        t.Name,
					Description: sdk.String(t.Description),
					Parameters:  schemaMap,
				},
			}
			params.Tools = append(params.Tools, toolParam)
		}
	} else if options.ResponseSchema != nil {
		p.logger.Info("[OpenAI] Using Structured Output (JSON Schema) mode.")
		schemaMap, err := llm.ConvertSchemaToMap(options.ResponseSchema)
		if err != nil {
			p.logger.Error("[OpenAI] Failed to convert ResponseSchema to map: ", err)
			return "", nil, errors.New("failed to process response schema")
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
	}

	resp, err := p.client.Chat.Completions.New(ctx, params)
	if err != nil {
		p.logger.Error("[OpenAI] API error: ", err)
		return "", nil, err
	}

	if len(resp.Choices) == 0 {
		p.logger.Warning("[OpenAI] No choices returned from API")
		return "", nil, nil
	}

	choice := resp.Choices[0]
	responseText := ""
	usage := &llm.UsageInfo{}

	usage.InputTokens = int(resp.Usage.PromptTokens)
	usage.OutputTokens = int(resp.Usage.CompletionTokens)

	if len(choice.Message.ToolCalls) > 0 {
		p.logger.Infof("[OpenAI] Received %d Tool Call(s). Returning arguments of the first call.", len(choice.Message.ToolCalls))
		responseText = choice.Message.ToolCalls[0].Function.Arguments
	} else {
		responseText = choice.Message.Content
	}

	if resp.SystemFingerprint != "" {
		p.logger.Info("[OpenAI] System Fingerprint: " + resp.SystemFingerprint)
	}

	return responseText, usage, nil
}

// GenerateTextStream generates a response piece by piece using SSE.
func (p *Provider) GenerateTextStream(ctx context.Context, prompt string, outChan chan<- llm.StreamChunk, opts ...llm.GenerationOption) (*llm.UsageInfo, error) {
	defer close(outChan)

	options := &llm.GenerationOptions{
		Temperature: llm.ValuePtr(float32(0.7)),
		MaxTokens:   llm.ValuePtr(int32(1024)),
	}
	for _, opt := range opts {
		opt(options)
	}

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

	messages := []sdk.ChatCompletionMessageParamUnion{}
	if systemPrompt != "" {
		messages = append(messages, sdk.SystemMessage(systemPrompt))
	}
	messages = append(messages, sdk.UserMessage(prompt))

	params := sdk.ChatCompletionNewParams{
		Messages: messages,
		Model:    p.modelName,
	}

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
		p.logger.Warning("[OpenAI Stream] Tool Calling (WithTools) is disabled for streaming in this provider. Ignoring tools.")
	} else if options.ResponseSchema != nil {
		p.logger.Warning("[OpenAI Stream] Structured Output (WithResponseSchema) is not supported for streaming by OpenAI. Ignoring schema.")
	}

	stream := p.client.Chat.Completions.NewStreaming(ctx, params)
	defer stream.Close()

	var lastUsage *sdk.CompletionUsage
	var systemFingerprint string

	for stream.Next() {
		resp := stream.Current()

		if len(resp.Choices) > 0 {
			deltaContent := resp.Choices[0].Delta.Content
			if deltaContent != "" {
				select {
				case outChan <- llm.StreamChunk{Delta: deltaContent}:
				case <-ctx.Done():
					p.logger.Info("[OpenAI Stream] Context cancelled during send.")
					finalUsageInfo, _ := processFinalUsage(lastUsage, p.logger)
					return finalUsageInfo, ctx.Err()
				}
			}
		}

		lastUsage = &resp.Usage

		if resp.SystemFingerprint != "" {
			systemFingerprint = resp.SystemFingerprint
		}
	}

	streamErr := stream.Err()
	if streamErr != nil && !errors.Is(streamErr, io.EOF) {
		p.logger.Error("[OpenAI Stream] Stream error: ", streamErr)
		finalUsageInfo, procErr := processFinalUsage(lastUsage, p.logger)
		if procErr != nil {
			p.logger.Errorf("[OpenAI Stream] Error processing usage data after stream error: %v", procErr)
		}
		return finalUsageInfo, streamErr
	}

	finalUsageInfo, procErr := processFinalUsage(lastUsage, p.logger)
	if procErr != nil {
		p.logger.Errorf("[OpenAI Stream] Error processing final usage data: %v", procErr)
	}

	if systemFingerprint != "" {
		p.logger.Info("[OpenAI Stream] System Fingerprint: " + systemFingerprint)
	}

	return finalUsageInfo, nil
}

func processFinalUsage(lastUsage *sdk.CompletionUsage, log logger.Logger) (*llm.UsageInfo, error) {
	if lastUsage == nil {
		log.Warning("[OpenAI Stream] No usage information received during stream.")
		return nil, nil
	}

	log.Info("[OpenAI Stream] Processing final usage data.")
	finalUsageInfo := &llm.UsageInfo{
		InputTokens:  int(lastUsage.PromptTokens),
		OutputTokens: int(lastUsage.CompletionTokens),
		CacheHitTokens: 0,
	}

	cacheHitTokens := int(lastUsage.PromptTokensDetails.CachedTokens)
	if cacheHitTokens > 0 {
		log.Infof("[OpenAI Stream] Final Cache hit: %d tokens", cacheHitTokens)
	}

	finalUsageInfo.CacheHitTokens = cacheHitTokens

	return finalUsageInfo, nil
}

// Close cleans up resources.
func (p *Provider) Close() error {
	p.logger.Info("[OpenAI] Provider closed.")
	return nil
}
