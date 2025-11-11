# go-multi-llm

A unified Go interface for multiple Large Language Model (LLM) providers with streaming support, structured logging, and comprehensive error handling.

## Features

- **Multi-Provider Support**: Unified interface for OpenAI, Claude, Gemini, Groq, DeepSeek, Cerebras, AI302, Sambanova, and more
- **Streaming Responses**: Real-time streaming support for all providers
- **Structured Logging**: Built-in logging with slog
- **Type-Safe**: Comprehensive type definitions for requests and responses
- **Error Handling**: Consistent error handling across all providers
- **Tool/Function Calling**: Support for function calling where available
- **JSON Mode**: Structured output support for compatible providers

## Supported Providers

| Provider | Streaming | Function Calling | JSON Mode |
|----------|-----------|------------------|-----------|
| OpenAI | ✅ | ✅ | ✅ |
| Claude (Anthropic) | ✅ | ✅ | ✅ |
| Google Gemini | ✅ | ✅ | ✅ |
| Groq | ✅ | ✅ | ✅ |
| DeepSeek | ✅ | ✅ | ✅ |
| Cerebras | ✅ | ❌ | ✅ |
| AI302 | ✅ | ❌ | ✅ |
| Sambanova | ✅ | ❌ | ✅ |

## Installation

```bash
go get github.com/ulgerang/go-multi-llm
```

## Quick Start

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/ulgerang/go-multi-llm/llm"
    "github.com/ulgerang/go-multi-llm/providers/openai"
)

func main() {
    ctx := context.Background()

    // Create provider
    provider, err := openai.NewProvider(openai.Config{
        APIKey: "your-api-key",
        Model:  "gpt-4-turbo-preview",
    })
    if err != nil {
        log.Fatal(err)
    }

    // Simple chat
    messages := []llm.Message{
        {Role: "user", Content: "Hello, how are you?"},
    }

    response, err := provider.ChatCompletion(ctx, messages, nil)
    if err != nil {
        log.Fatal(err)
    }

    fmt.Println(response.Content)
}
```

## Streaming Example

```go
func streamingExample() {
    provider, _ := openai.NewProvider(openai.Config{
        APIKey: "your-api-key",
        Model:  "gpt-4-turbo-preview",
    })

    messages := []llm.Message{
        {Role: "user", Content: "Write a short poem"},
    }

    stream, err := provider.ChatCompletionStream(ctx, messages, nil)
    if err != nil {
        log.Fatal(err)
    }
    defer stream.Close()

    for {
        chunk, err := stream.Recv()
        if err == io.EOF {
            break
        }
        if err != nil {
            log.Fatal(err)
        }

        fmt.Print(chunk.Content)
    }
}
```

## Function Calling Example

```go
func functionCallingExample() {
    tools := []llm.Tool{
        {
            Type: "function",
            Function: llm.Function{
                Name:        "get_weather",
                Description: "Get current weather for a location",
                Parameters: map[string]interface{}{
                    "type": "object",
                    "properties": map[string]interface{}{
                        "location": map[string]interface{}{
                            "type":        "string",
                            "description": "City name",
                        },
                    },
                    "required": []string{"location"},
                },
            },
        },
    }

    options := &llm.ChatOptions{
        Tools: tools,
    }

    messages := []llm.Message{
        {Role: "user", Content: "What's the weather in Seoul?"},
    }

    response, _ := provider.ChatCompletion(ctx, messages, options)

    if len(response.ToolCalls) > 0 {
        fmt.Printf("Function to call: %s\n", response.ToolCalls[0].Function.Name)
        fmt.Printf("Arguments: %s\n", response.ToolCalls[0].Function.Arguments)
    }
}
```

## Configuration

Each provider can be configured with specific options:

```go
// OpenAI
openaiProvider, _ := openai.NewProvider(openai.Config{
    APIKey:      "your-key",
    Model:       "gpt-4-turbo-preview",
    BaseURL:     "https://api.openai.com/v1", // optional
    Temperature: 0.7,
})

// Claude
claudeProvider, _ := claude.NewProvider(claude.Config{
    APIKey:  "your-key",
    Model:   "claude-3-opus-20240229",
    Version: "2023-06-01", // API version
})

// Gemini
geminiProvider, _ := gemini.NewProvider(gemini.Config{
    APIKey: "your-key",
    Model:  "gemini-1.5-pro",
})
```

## Environment Variables

You can use environment variables for API keys:

```bash
export OPENAI_API_KEY=your-openai-key
export CLAUDE_API_KEY=your-claude-key
export GEMINI_API_KEY=your-gemini-key
export GROQ_API_KEY=your-groq-key
export DEEPSEEK_API_KEY=your-deepseek-key
export CEREBRAS_API_KEY=your-cerebras-key
export AI302_API_KEY=your-ai302-key
export SAMBANOVA_API_KEY=your-sambanova-key
```

Then create providers without specifying the API key:

```go
provider, _ := openai.NewProvider(openai.Config{
    Model: "gpt-4-turbo-preview",
    // APIKey will be read from OPENAI_API_KEY env var
})
```

## Provider Interface

All providers implement the `llm.Provider` interface:

```go
type Provider interface {
    ChatCompletion(ctx context.Context, messages []Message, options *ChatOptions) (*ChatResponse, error)
    ChatCompletionStream(ctx context.Context, messages []Message, options *ChatOptions) (Stream, error)
    GetModelName() string
}
```

## Logging

The package uses structured logging with `slog`. You can configure the logger:

```go
import "github.com/ulgerang/go-multi-llm/logger"

// Set log level
logger.SetLevel(slog.LevelDebug)

// Use custom logger
customLogger := slog.New(slog.NewJSONHandler(os.Stdout, nil))
logger.SetLogger(customLogger)
```

## Error Handling

Errors are wrapped with context information:

```go
response, err := provider.ChatCompletion(ctx, messages, nil)
if err != nil {
    // Check for specific error types
    if errors.Is(err, context.DeadlineExceeded) {
        log.Println("Request timed out")
    } else {
        log.Printf("API error: %v", err)
    }
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details
