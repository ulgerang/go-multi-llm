package zai

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestChatRequestMarshaling(t *testing.T) {
	temp := 1.0
	maxTokens := int64(4096)
	req := ChatRequest{
		Model: "glm-4.7",
		Messages: []ChatMessage{
			{Role: "user", Content: "Hello"},
		},
		Temperature: &temp,
		MaxTokens:   &maxTokens,
		Thinking:    &ThinkingConfig{Type: "enabled"},
	}

	body, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("failed to marshal request: %v", err)
	}

	jsonStr := string(body)
	t.Logf("JSON: %s", jsonStr)

	if !strings.Contains(jsonStr, `"thinking":{"type":"enabled"}`) {
		t.Errorf("expected thinking parameter not found in JSON: %s", jsonStr)
	}
	if !strings.Contains(jsonStr, `"temperature":1`) {
		t.Errorf("expected temperature parameter not found in JSON: %s", jsonStr)
	}
	if !strings.Contains(jsonStr, `"max_tokens":4096`) {
		t.Errorf("expected max_tokens parameter not found in JSON: %s", jsonStr)
	}
}
