# modules/go-multi-llm - Multi-Provider LLM Client

**Generated:** 2026-01-17T09:45:00+0900

## OVERVIEW
Unified Go interface for multiple LLM providers with streaming, function calling, and structured output.

## STRUCTURE
```
modules/go-multi-llm/
├── llm/
│   ├── provider.go      # Provider interface
│   ├── types.go         # GenerationOptions, StreamChunk, Tool
│   └── schema.go        # JSON Schema for structured output
├── providers/
│   ├── openai/          # OpenAI (GPT-4, etc.)
│   ├── claude/          # Anthropic Claude
│   ├── gemini/          # Google Gemini
│   ├── groq/            # Groq
│   ├── deepseek/        # DeepSeek
│   ├── cerebras/        # Cerebras
│   ├── ai302/           # AI302
│   ├── grok/            # xAI Grok
│   ├── openrouter/      # OpenRouter aggregator
│   ├── inception/       # Inception
│   └── zai/             # ZAI
├── logger/              # Structured logging
└── utils/               # JSON utilities
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| **Provider interface** | `llm/provider.go` | `ChatCompletion()`, `ChatCompletionStream()` |
| **Options** | `llm/types.go` | `GenerationOptions`, `WithTemperature()`, etc. |
| **Tool calling** | `llm/types.go` | `Tool` struct with JSON Schema |
| **Add provider** | `providers/*/provider.go` | Each provider in own package |

## CONVENTIONS
- **Functional options**: `WithTemperature(0.7)`, `WithMaxTokens(1000)`
- **Provider config**: Each provider has own `Config` struct
- **Streaming**: Return `<-chan StreamChunk` for streaming responses
- **Error handling**: Wrap HTTP errors with context

## ANTI-PATTERNS
- **Never**: Hardcode API keys (use env vars or config)
- **Never**: Ignore streaming errors (check `chunk.Err`)
- **Never**: Skip `Close()` on stream channels

## USAGE
```go
provider, _ := openai.NewProvider(openai.Config{Model: "gpt-4o"})
response, _ := provider.ChatCompletion(ctx, messages, &llm.ChatOptions{Tools: tools})
```
