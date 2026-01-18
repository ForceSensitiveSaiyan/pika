@echo off
REM Run PIKA tests with coverage

echo Running PIKA Tests
echo ================================

REM Check if pytest is available
where pytest >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo pytest not found. Installing dev dependencies...
    pip install -e ".[dev]"
)

echo.
echo Running tests with coverage...
echo.

pytest tests/ ^
    --cov=src/pika ^
    --cov-report=term-missing ^
    --cov-report=html:coverage_html ^
    --cov-fail-under=50 ^
    -v ^
    %*

if %ERRORLEVEL% EQU 0 (
    echo.
    echo All tests passed!
    echo Coverage report: coverage_html\index.html
) else (
    echo.
    echo Tests failed!
)

exit /b %ERRORLEVEL%
