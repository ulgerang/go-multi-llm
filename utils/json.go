package utils

import (
	"encoding/json"
	"errors"
	"regexp"
	"strings"
)

var languageNames = map[string]string{
	"en": "English",
	"ko": "Korean",
	"ja": "Japanese",
	"zh": "Chinese",
	"fr": "French",
	"de": "German",
	"es": "Spanish",
	"it": "Italian",
	"ru": "Russian",
	"pt": "Portuguese",
}

// GetLangName returns the human-readable name for a language code. If unknown, the code itself is returned.
func GetLangName(code string) string {
	if name, ok := languageNames[code]; ok {
		return name
	}
	return code
}

// ExtractJSONFromString attempts to extract the first valid JSON payload from a string.
// It supports markdown ```json code fences``` and raw JSON strings.
func ExtractJSONFromString(s string) (string, error) {
	re := regexp.MustCompile("(?s)```json\\s*(.*?)\\s*```")
	matches := re.FindStringSubmatch(s)

	if len(matches) > 1 {
		jsonStr := strings.TrimSpace(matches[1])
		var js json.RawMessage
		if err := json.Unmarshal([]byte(jsonStr), &js); err == nil {
			return jsonStr, nil
		}
	}

	trimmed := strings.TrimSpace(s)
	if (strings.HasPrefix(trimmed, "{") && strings.HasSuffix(trimmed, "}")) ||
		(strings.HasPrefix(trimmed, "[") && strings.HasSuffix(trimmed, "]")) {
		var js json.RawMessage
		if err := json.Unmarshal([]byte(trimmed), &js); err == nil {
			return trimmed, nil
		}
	}

	return "", errors.New("no valid JSON found in the string or markdown block")
}

// ExtractValidJSON returns the most likely JSON fragment from a free-form string.
// If no valid JSON is found, the trimmed original string is returned.
func ExtractValidJSON(raw string) string {
	raw = strings.TrimSpace(raw)

	backtickCount := strings.Count(raw, "`")
	if backtickCount == 1 {
		raw = strings.ReplaceAll(raw, "`", "")
	}

	if strings.HasPrefix(raw, "```json") && strings.HasSuffix(raw, "```") {
		raw = strings.TrimPrefix(raw, "```json")
		raw = strings.TrimSuffix(raw, "```")
		raw = strings.TrimSpace(raw)
	} else if strings.HasPrefix(raw, "```") && strings.HasSuffix(raw, "```") {
		raw = strings.TrimPrefix(raw, "```")
		raw = strings.TrimSuffix(raw, "```")
		raw = strings.TrimSpace(raw)
	}

	firstBrace := strings.Index(raw, "{")
	firstBracket := strings.Index(raw, "[")
	lastBrace := strings.LastIndex(raw, "}")
	lastBracket := strings.LastIndex(raw, "]")

	start := -1
	end := -1

	if firstBrace != -1 && lastBrace != -1 && lastBrace > firstBrace {
		if firstBracket == -1 || (firstBrace < firstBracket && lastBrace > lastBracket) {
			start = firstBrace
			end = lastBrace + 1
		}
	}

	if firstBracket != -1 && lastBracket != -1 && lastBracket > firstBracket {
		if firstBrace == -1 || firstBracket < firstBrace || start == -1 {
			start = firstBracket
			end = lastBracket + 1
		}
	}

	if start != -1 && end != -1 && end > start {
		candidate := raw[start:end]
		var js json.RawMessage
		if json.Unmarshal([]byte(candidate), &js) == nil {
			return candidate
		}
	}

	start = -1
	if firstBrace != -1 {
		start = firstBrace
	}
	if firstBracket != -1 && (start == -1 || firstBracket < start) {
		start = firstBracket
	}

	if start != -1 {
		actualEnd := -1
		switch raw[start] {
		case '{':
			if lastBrace > start {
				actualEnd = lastBrace + 1
			}
		case '[':
			if lastBracket > start {
				actualEnd = lastBracket + 1
			}
		}

		if actualEnd != -1 && actualEnd <= len(raw) {
			candidate := raw[start:actualEnd]
			var js json.RawMessage
			if json.Unmarshal([]byte(candidate), &js) == nil {
				return candidate
			}
		}
	}

	return raw
}

// SafeUnmarshalJSON extracts JSON from a string and unmarshals it into the provided destination.
func SafeUnmarshalJSON(data string, v interface{}) error {
	extracted := ExtractValidJSON(data)
	return json.Unmarshal([]byte(extracted), v)
}
