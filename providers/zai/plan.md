# Plan: Implement Z.AI Thinking Parameter

## Steps
1. **Modify `provider.go` Structures**:
   - Add `ThinkingConfig` struct:
     ```go
     type ThinkingConfig struct {
         Type string `json:"type"`
     }
     ```
   - Update `ChatRequest` to include `Thinking *ThinkingConfig `json:"thinking,omitempty"``.

2. **Update `GenerateText`**:
   - Initialize `Thinking` field in `ChatRequest` with `type: "enabled"`.

3. **Update `GenerateTextStream`**:
   - Initialize `Thinking` field in `ChatRequest` with `type: "enabled"`.

4. **Verify Implementation**:
   - Check if any other fields need to be updated (e.g., `MaxTokens`, `Temperature` were commented out, maybe the user wants them back in if `thinking` is enabled).
   - The user provided `max_tokens: 4096` and `temperature: 1.0` in the curl. I should probably ensure these are handled if provided.

## TDD (Red-Green-Refactor)
- I'll look for existing tests for this provider to see how to add a test case.
