package llm

// ProviderType represents the type of LLM provider.
type ProviderType string

const (
	OpenAIProviderType      ProviderType = "openai"
	GeminiProviderType      ProviderType = "gemini"
	SambanovaProviderType   ProviderType = "sambanova"
	ClaudeProviderType      ProviderType = "claude"
	GroqProviderType        ProviderType = "groq"
	GrokProviderType        ProviderType = "grok"
	DeepSeekProviderType    ProviderType = "deepseek"
	AI302ProviderType       ProviderType = "ai302"
	SolarProviderType       ProviderType = "solar"
	InceptionProviderType   ProviderType = "inception"
	TrillionProviderType    ProviderType = "trillion"
	OpenRouterProviderType  ProviderType = "openrouter"
	CerebrasProviderType    ProviderType = "cerebras"
)
