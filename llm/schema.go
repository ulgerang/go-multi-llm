package llm

import "encoding/json"

// SchemaProperty represents a property in a JSON schema with extended features.
type SchemaProperty struct {
	Type                 string                     `json:"type,omitempty"`
	Description          string                     `json:"description,omitempty"`
	Format               string                     `json:"format,omitempty"`
	Properties           map[string]*SchemaProperty `json:"properties,omitempty"`
	Items                *SchemaProperty            `json:"items,omitempty"`
	MinLength            *int                       `json:"minLength,omitempty"`
	MaxLength            *int                       `json:"maxLength,omitempty"`
	Pattern              string                     `json:"pattern,omitempty"`
	Minimum              *float64                   `json:"minimum,omitempty"`
	Maximum              *float64                   `json:"maximum,omitempty"`
	MultipleOf           *float64                   `json:"multipleOf,omitempty"`
	MinItems             *int                       `json:"minItems,omitempty"`
	MaxItems             *int                       `json:"maxItems,omitempty"`
	UniqueItems          bool                       `json:"uniqueItems,omitempty"`
	Required             []string                   `json:"required,omitempty"`
	Enum                 []interface{}              `json:"enum,omitempty"`
	Const                interface{}                `json:"const,omitempty"`
	Default              interface{}                `json:"default,omitempty"`
	Ref                  string                     `json:"$ref,omitempty"`
	AdditionalProperties *bool                      `json:"additionalProperties,omitempty"`
}

// ConvertToJSONSchema converts a SchemaProperty into a JSON schema string.
func ConvertToJSONSchema(property *SchemaProperty) (string, error) {
	schemaMap, err := ConvertSchemaToMap(property)
	if err != nil {
		return "", err
	}

	jsonBytes, err := json.Marshal(schemaMap)
	if err != nil {
		return "", err
	}

	return string(jsonBytes), nil
}

// ConvertSchemaToMap converts a SchemaProperty into a generic map representation.
func ConvertSchemaToMap(property *SchemaProperty) (map[string]interface{}, error) {
	schemaMap := make(map[string]interface{})

	if property.Type != "" {
		schemaMap["type"] = property.Type
	}

	if property.Properties != nil {
		props := make(map[string]interface{})
		for key, prop := range property.Properties {
			subSchema, err := ConvertSchemaToMap(prop)
			if err != nil {
				return nil, err
			}
			props[key] = subSchema
		}
		schemaMap["properties"] = props
	}

	if property.Items != nil {
		subSchema, err := ConvertSchemaToMap(property.Items)
		if err != nil {
			return nil, err
		}
		schemaMap["items"] = subSchema
	}

	if property.MinLength != nil {
		schemaMap["minLength"] = *property.MinLength
	}

	if property.MaxLength != nil {
		schemaMap["maxLength"] = *property.MaxLength
	}

	if property.Pattern != "" {
		schemaMap["pattern"] = property.Pattern
	}

	if property.Minimum != nil {
		schemaMap["minimum"] = *property.Minimum
	}

	if property.Maximum != nil {
		schemaMap["maximum"] = *property.Maximum
	}

	if property.MultipleOf != nil {
		schemaMap["multipleOf"] = *property.MultipleOf
	}

	if property.MinItems != nil {
		schemaMap["minItems"] = *property.MinItems
	}

	if property.MaxItems != nil {
		schemaMap["maxItems"] = *property.MaxItems
	}

	if property.UniqueItems {
		schemaMap["uniqueItems"] = property.UniqueItems
	}

	if len(property.Required) > 0 {
		schemaMap["required"] = property.Required
	}

	if len(property.Enum) > 0 {
		schemaMap["enum"] = property.Enum
	}

	if property.Const != nil {
		schemaMap["const"] = property.Const
	}

	if property.Default != nil {
		schemaMap["default"] = property.Default
	}

	if property.Ref != "" {
		schemaMap["$ref"] = property.Ref
	}

	if property.AdditionalProperties != nil {
		schemaMap["additionalProperties"] = *property.AdditionalProperties
	}

	return schemaMap, nil
}
