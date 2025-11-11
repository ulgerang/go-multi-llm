package logger

// Logger defines the minimal logging interface expected by llm-module providers.
type Logger interface {
	Debug(message string)
	Debugf(format string, args ...interface{})
	Info(message string)
	Infof(format string, args ...interface{})
	Warning(message string)
	Warningf(format string, args ...interface{})
	Error(message string, err error)
	Errorf(format string, args ...interface{})
}
