package zai

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/ulgerang/llm-module/llm"
	"github.com/ulgerang/llm-module/testutil"
)

// testLogger implements logger.Logger interface for testing
type testLogger struct {
	t *testing.T
}

func (l *testLogger) Debug(message string)                      { l.t.Log("[DEBUG]", message) }
func (l *testLogger) Debugf(format string, args ...interface{}) { l.t.Logf("[DEBUG] "+format, args...) }
func (l *testLogger) Info(message string)                       { l.t.Log("[INFO]", message) }
func (l *testLogger) Infof(format string, args ...interface{})  { l.t.Logf("[INFO] "+format, args...) }
func (l *testLogger) Warning(message string)                    { l.t.Log("[WARN]", message) }
func (l *testLogger) Warningf(format string, args ...interface{}) {
	l.t.Logf("[WARN] "+format, args...)
}
func (l *testLogger) Error(message string, err error)           { l.t.Log("[ERROR]", message, err) }
func (l *testLogger) Errorf(format string, args ...interface{}) { l.t.Logf("[ERROR] "+format, args...) }

// TestZAIReasoningContentBug tests if the provider correctly separates
// reasoning_content from content when thinking mode is enabled.
//
// Expected behavior:
// - reasoning_content: Contains LLM's internal reasoning process (should NOT be returned)
// - content: Contains the actual response (should be returned)
//
// Bug symptom:
// - When content is empty, the provider incorrectly returns reasoning_content
// - This causes LLM "thinking" to appear as actual output
func TestZAIReasoningContentBug(t *testing.T) {
	apiKey := testutil.SkipIfNoAPIKey(t, "zai")

	log := &testLogger{t: t}
	provider, err := New(log, apiKey, "glm-4.7")
	if err != nil {
		t.Fatalf("Failed to create provider: %v", err)
	}
	defer provider.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	// Simple test that should produce a clear, short response
	prompt := "What is 2 + 2? Answer with just the number."

	result, usage, err := provider.GenerateText(ctx, prompt,
		llm.WithTemperature(0.1),
		llm.WithMaxTokens(100), // Intentionally low to test token budget issue
	)

	if err != nil {
		t.Fatalf("GenerateText failed: %v", err)
	}

	t.Logf("Result: %q", result)
	t.Logf("Usage: %+v", usage)

	// Check for reasoning content leakage patterns
	reasoningPatterns := []string{
		"The user wants",
		"Let me think",
		"I need to",
		"First, I'll",
		"The question asks",
		"To answer this",
	}

	for _, pattern := range reasoningPatterns {
		if contains(result, pattern) {
			t.Errorf("REASONING CONTENT LEAKAGE DETECTED: result contains '%s'", pattern)
			t.Errorf("Full result: %s", result)
		}
	}

	// The result should be very short for this simple question
	if len(result) > 50 {
		t.Logf("WARNING: Result unusually long (%d chars), may contain reasoning", len(result))
		t.Logf("Result: %s", result)
	}
}

// TestZAIThinkingModeTokenBudget tests if the token budget is sufficient
// for both reasoning and content generation.
func TestZAIThinkingModeTokenBudget(t *testing.T) {
	apiKey := testutil.SkipIfNoAPIKey(t, "zai")

	log := &testLogger{t: t}
	provider, err := New(log, apiKey, "glm-4.7")
	if err != nil {
		t.Fatalf("Failed to create provider: %v", err)
	}
	defer provider.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	// Request that requires actual content generation
	prompt := `Generate a simple Go function that adds two numbers.
Output ONLY the code, no explanations.`

	result, usage, err := provider.GenerateText(ctx, prompt,
		llm.WithTemperature(0.2),
	)

	if err != nil {
		t.Fatalf("GenerateText failed: %v", err)
	}

	t.Logf("Result length: %d chars", len(result))
	t.Logf("Usage: %+v", usage)

	// Check if result looks like actual code, not reasoning
	if !contains(result, "func") {
		t.Errorf("Result doesn't look like Go code")
		t.Errorf("Result: %s", result)
	}

	// Check for reasoning content patterns
	if contains(result, "The user wants") || contains(result, "Let me") {
		t.Errorf("Result contains reasoning content instead of code")
		t.Errorf("Result: %s", result)
	}
}

// TestZAIStreamingReasoningContent tests streaming mode for the same bug.
func TestZAIStreamingReasoningContent(t *testing.T) {
	apiKey := testutil.SkipIfNoAPIKey(t, "zai")

	log := &testLogger{t: t}
	provider, err := New(log, apiKey, "glm-4.7")
	if err != nil {
		t.Fatalf("Failed to create provider: %v", err)
	}
	defer provider.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	prompt := "What is the capital of France? Answer in one word."

	outChan := make(chan llm.StreamChunk, 100)

	go func() {
		_, err := provider.GenerateTextStream(ctx, prompt, outChan,
			llm.WithTemperature(0.1),
			llm.WithMaxTokens(100),
		)
		if err != nil {
			t.Errorf("GenerateTextStream failed: %v", err)
		}
	}()

	var fullResult string
	var reasoningDetected bool

	for chunk := range outChan {
		if chunk.Err != nil {
			t.Fatalf("Stream error: %v", chunk.Err)
		}
		if chunk.IsFinal {
			break
		}

		fullResult += chunk.Delta

		// Check each chunk for reasoning patterns
		if contains(chunk.Delta, "The user") || contains(chunk.Delta, "Let me") {
			reasoningDetected = true
			t.Logf("Reasoning detected in chunk: %q", chunk.Delta)
		}
	}

	t.Logf("Full streaming result: %q", fullResult)

	if reasoningDetected {
		t.Errorf("STREAMING REASONING CONTENT LEAKAGE DETECTED")
	}
}

// TestZAIResponseFields directly tests the API response structure.
func TestZAIResponseFields(t *testing.T) {
	apiKey := testutil.SkipIfNoAPIKey(t, "zai")

	log := &testLogger{t: t}
	provider, err := New(log, apiKey, "glm-4.7")
	if err != nil {
		t.Fatalf("Failed to create provider: %v", err)
	}
	defer provider.Close()

	// This test would need access to the raw response to check both fields
	// For now, we document the expected behavior:
	//
	// Z.AI API Response with thinking enabled:
	// {
	//   "choices": [{
	//     "message": {
	//       "role": "assistant",
	//       "content": "4",                           // <-- This should be returned
	//       "reasoning_content": "The user asked..."  // <-- This should be IGNORED
	//     }
	//   }]
	// }
	//
	// Current bug:
	// When content is empty, reasoning_content is returned instead.

	t.Log("This test documents the expected API behavior.")
	t.Log("See provider.go lines 293-297 and 457-460 for the bug location.")
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > 0 && findSubstring(s, substr))
}

func findSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

// BenchmarkZAIThinkingOverhead measures the overhead of thinking mode.
func BenchmarkZAIThinkingOverhead(b *testing.B) {
	apiKey := testutil.MustGetAPIKey("zai")

	// benchLogger for benchmarks (no testing.T available)
	log := &benchLogger{}
	provider, err := New(log, apiKey, "glm-4.7")
	if err != nil {
		b.Fatalf("Failed to create provider: %v", err)
	}
	defer provider.Close()

	ctx := context.Background()
	prompt := "What is 1+1?"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, err := provider.GenerateText(ctx, prompt, llm.WithMaxTokens(50))
		if err != nil {
			b.Errorf("Request %d failed: %v", i, err)
		}
	}
}

// benchLogger implements logger.Logger for benchmarks (silent)
type benchLogger struct{}

func (l *benchLogger) Debug(message string)                        {}
func (l *benchLogger) Debugf(format string, args ...interface{})   {}
func (l *benchLogger) Info(message string)                         {}
func (l *benchLogger) Infof(format string, args ...interface{})    {}
func (l *benchLogger) Warning(message string)                      {}
func (l *benchLogger) Warningf(format string, args ...interface{}) {}
func (l *benchLogger) Error(message string, err error)             {}
func (l *benchLogger) Errorf(format string, args ...interface{})   {}

func init() {
	// Print test info
	fmt.Println("=== Z.AI Provider Smoke Test ===")
	fmt.Println("Testing for ReasoningContent leakage bug")
	fmt.Println("See: modules/go-multi-llm/providers/zai/provider.go:293-297, 457-460")
	fmt.Println("")
}
