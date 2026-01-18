// Package testutil provides testing utilities for LLM provider smoke tests.
//
// This package loads API keys and model configurations from ~/.holon/providers.yaml,
// allowing smoke tests to run without hardcoded credentials or environment variables.
//
// Usage:
//
//	apiKey := testutil.GetAPIKey("zai")
//	if apiKey == "" {
//	    t.Skip("ZAI API key not configured")
//	}
//
//	config := testutil.GetProviderConfig("openai")
//	provider, _ := openai.NewProvider(openai.Config{
//	    APIKey: config.APIKey,
//	    Model:  config.DefaultModel,
//	})
package testutil

import (
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"

	"gopkg.in/yaml.v3"
)

// ProviderConfig holds configuration for a single LLM provider.
type ProviderConfig struct {
	APIKey       string `yaml:"api_key"`
	APIKeyFile   string `yaml:"api_key_file"`
	DefaultModel string `yaml:"default_model"`
	BaseURL      string `yaml:"base_url"`
}

// ProvidersConfig represents the ~/.holon/providers.yaml structure.
type ProvidersConfig struct {
	Default   string                     `yaml:"default"`
	Providers map[string]*ProviderConfig `yaml:"providers"`
}

var (
	cachedConfig *ProvidersConfig
	envVarRe     = regexp.MustCompile(`\$\{([^}]+)\}`)
)

// wellKnownEnvVars maps provider names to their well-known environment variables.
var wellKnownEnvVars = map[string]string{
	"openai":     "OPENAI_API_KEY",
	"anthropic":  "ANTHROPIC_API_KEY",
	"claude":     "CLAUDE_API_KEY",
	"openrouter": "OPENROUTER_API_KEY",
	"gemini":     "GEMINI_API_KEY",
	"google":     "GEMINI_API_KEY",
	"deepseek":   "DEEPSEEK_API_KEY",
	"groq":       "GROQ_API_KEY",
	"zai":        "ZAI_API_KEY",
	"grok":       "GROK_API_KEY",
	"cerebras":   "CEREBRAS_API_KEY",
	"ai302":      "AI302_API_KEY",
}

// GetHomeDir returns the user's home directory.
func GetHomeDir() string {
	if runtime.GOOS == "windows" {
		if home := os.Getenv("USERPROFILE"); home != "" {
			return home
		}
		return os.Getenv("HOMEDRIVE") + os.Getenv("HOMEPATH")
	}
	return os.Getenv("HOME")
}

// GetHolonConfigDir returns the path to ~/.holon directory.
func GetHolonConfigDir() string {
	home := GetHomeDir()
	if home == "" {
		return ""
	}
	return filepath.Join(home, ".holon")
}

// LoadProvidersConfig loads and caches the providers.yaml configuration.
// Returns nil if the config file doesn't exist or can't be parsed.
func LoadProvidersConfig() *ProvidersConfig {
	if cachedConfig != nil {
		return cachedConfig
	}

	configDir := GetHolonConfigDir()
	if configDir == "" {
		return nil
	}

	configPath := filepath.Join(configDir, "providers.yaml")
	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil
	}

	var config ProvidersConfig
	if err := yaml.Unmarshal(data, &config); err != nil {
		return nil
	}

	cachedConfig = &config
	return cachedConfig
}

// GetProviderConfig returns the configuration for a specific provider.
// Returns nil if the provider is not configured.
func GetProviderConfig(providerName string) *ProviderConfig {
	config := LoadProvidersConfig()
	if config == nil || config.Providers == nil {
		return nil
	}

	providerConfig, ok := config.Providers[strings.ToLower(providerName)]
	if !ok {
		return nil
	}
	return providerConfig
}

// GetAPIKey returns the API key for a provider.
// Resolution order:
//  1. Environment variable (e.g., ZAI_API_KEY)
//  2. ~/.holon/providers.yaml (with ${ENV_VAR} expansion)
//  3. API key file (if specified in config)
//
// Returns empty string if no key is found.
func GetAPIKey(providerName string) string {
	providerName = strings.ToLower(providerName)

	// 1. Try well-known environment variable first
	if envVar, ok := wellKnownEnvVars[providerName]; ok {
		if key := os.Getenv(envVar); key != "" {
			return key
		}
	}

	// Also try generic pattern: PROVIDER_API_KEY
	genericEnvVar := strings.ToUpper(providerName) + "_API_KEY"
	if key := os.Getenv(genericEnvVar); key != "" {
		return key
	}

	// 2. Try providers.yaml
	providerConfig := GetProviderConfig(providerName)
	if providerConfig == nil {
		return ""
	}

	// Try API key file
	if providerConfig.APIKeyFile != "" {
		keyPath := expandPath(providerConfig.APIKeyFile)
		if data, err := os.ReadFile(keyPath); err == nil {
			return strings.TrimSpace(string(data))
		}
	}

	// Try API key with env var expansion
	apiKey := expandEnvVars(providerConfig.APIKey)
	return apiKey
}

// GetDefaultModel returns the default model for a provider.
func GetDefaultModel(providerName string) string {
	providerConfig := GetProviderConfig(providerName)
	if providerConfig == nil {
		return ""
	}
	return providerConfig.DefaultModel
}

// GetBaseURL returns the base URL for a provider.
func GetBaseURL(providerName string) string {
	providerConfig := GetProviderConfig(providerName)
	if providerConfig == nil {
		return ""
	}
	return expandEnvVars(providerConfig.BaseURL)
}

// expandEnvVars expands ${VAR_NAME} patterns in a string.
func expandEnvVars(s string) string {
	return envVarRe.ReplaceAllStringFunc(s, func(match string) string {
		varName := match[2 : len(match)-1]
		return os.Getenv(varName)
	})
}

// expandPath expands ~ to home directory.
func expandPath(path string) string {
	if strings.HasPrefix(path, "~/") {
		home := GetHomeDir()
		if home != "" {
			return filepath.Join(home, path[2:])
		}
	}
	return path
}

// SkipIfNoAPIKey is a test helper that skips the test if the API key is not available.
// Usage:
//
//	func TestSomething(t *testing.T) {
//	    apiKey := testutil.SkipIfNoAPIKey(t, "openai")
//	    // ... use apiKey
//	}
func SkipIfNoAPIKey(t interface{ Skip(args ...interface{}) }, providerName string) string {
	apiKey := GetAPIKey(providerName)
	if apiKey == "" {
		t.Skip(providerName + " API key not configured (set " +
			strings.ToUpper(providerName) + "_API_KEY or configure in ~/.holon/providers.yaml)")
	}
	return apiKey
}

// MustGetAPIKey panics if the API key is not available.
// Use this in benchmarks or non-test code.
func MustGetAPIKey(providerName string) string {
	apiKey := GetAPIKey(providerName)
	if apiKey == "" {
		panic(providerName + " API key not configured")
	}
	return apiKey
}
