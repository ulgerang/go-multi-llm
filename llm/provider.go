package llm

import "context"

// SystemBlock represents a block of text for the system prompt, potentially cacheable.
type SystemBlock struct {
	Text     string
	UseCache bool
}

// GenerationOptions holds all possible generation parameters
type GenerationOptions struct {
	Temperature        *float32
	MaxTokens          *int32
	TopK               *float32
	TopP               *float32
	Language           string
	System             string
	SystemBlocks       []SystemBlock
	ResponseFormat     string
	ResponseSchema     *SchemaProperty
	Tools              []*Tool
	UseCache           bool
	AllowSexualContent bool
}

// StreamChunk represents a piece of the streamed response.
type StreamChunk struct {
	Delta   string
	IsFinal bool
	Err     error
}

// Tool represents a function or capability the LLM can invoke.
type Tool struct {
	Name        string
	Description string
	InputSchema *SchemaProperty
}

func ValuePtr[T any](value T) *T {
	return &value
}

type GenerationOption func(options *GenerationOptions)

func WithTemperature(temp float32) GenerationOption {
	return func(options *GenerationOptions) {
		options.Temperature = ValuePtr(temp)
	}
}

func WithMaxTokens(tokens int32) GenerationOption {
	return func(options *GenerationOptions) {
		options.MaxTokens = ValuePtr(tokens)
	}
}

func WithTopK(topK float32) GenerationOption {
	return func(options *GenerationOptions) {
		options.TopK = ValuePtr(topK)
	}
}

func WithTopP(topP float32) GenerationOption {
	return func(options *GenerationOptions) {
		options.TopP = ValuePtr(topP)
	}
}

func WithLanguage(lang string) GenerationOption {
	return func(options *GenerationOptions) {
		options.Language = lang
	}
}

func WithSystem(system string) GenerationOption {
	return func(options *GenerationOptions) {
		options.System = system
	}
}

func WithResponseFormat(format string) GenerationOption {
	return func(options *GenerationOptions) {
		options.ResponseFormat = format
	}
}

func WithResponseSchema(schema *SchemaProperty) GenerationOption {
	return func(options *GenerationOptions) {
		options.ResponseSchema = schema
	}
}

func WithTools(tools []*Tool) GenerationOption {
	return func(options *GenerationOptions) {
		options.Tools = tools
	}
}

func WithSystemBlocks(blocks []SystemBlock) GenerationOption {
	return func(options *GenerationOptions) {
		options.SystemBlocks = blocks
	}
}

func WithCache(useCache bool) GenerationOption {
	return func(options *GenerationOptions) {
		options.UseCache = useCache
	}
}

func WithAllowSexualContent(allow bool) GenerationOption {
	return func(options *GenerationOptions) {
		options.AllowSexualContent = allow
	}
}

// UsageInfo contains token usage statistics
type UsageInfo struct {
	InputTokens       int
	OutputTokens      int
	CacheCreateTokens int
	CacheHitTokens    int
	CacheMissTokens   int
}

// Provider defines interface for LLM providers
type Provider interface {
	GenerateText(ctx context.Context, prompt string, options ...GenerationOption) (string, *UsageInfo, error)
	GenerateTextStream(ctx context.Context, prompt string, outChan chan<- StreamChunk, options ...GenerationOption) (*UsageInfo, error)
	GetModelName() string
	Close() error
}
