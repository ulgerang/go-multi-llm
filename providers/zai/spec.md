# Spec: Support Z.AI Thinking Parameter

## Overview
Added support for the `thinking` parameter in the Z.AI provider to enable reasoning capabilities in models like `glm-4.7`.

## Requirements (EARS)
- **THEN** the `ChatRequest` struct SHALL include a `Thinking` field that maps to the `thinking` JSON key.
- **WHEN** a request is sent to the Z.AI API, **THEN** the `thinking` parameter SHALL be included in the payload if enabled.
- **AS LONG AS** the model supports reasoning, **THEN** it SHALL be possible to enable it via the provider configuration or options.

## Implementation Details
- Define a `ThinkingConfig` struct with a `Type` field.
- Add `Thinking` field to `ChatRequest`.
- Default to enabling thinking if the model is `glm-4.7` or similar, or provide a way to toggle it.
- Ensure `GenerateText` and `GenerateTextStream` use this new field.
