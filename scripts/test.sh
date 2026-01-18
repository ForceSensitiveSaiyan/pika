#!/bin/bash
# Run PIKA tests with coverage

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Running PIKA Tests${NC}"
echo "================================"

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}pytest not found. Installing dev dependencies...${NC}"
    pip install -e ".[dev]"
fi

# Run tests with coverage
echo -e "\n${YELLOW}Running tests with coverage...${NC}\n"

pytest tests/ \
    --cov=src/pika \
    --cov-report=term-missing \
    --cov-report=html:coverage_html \
    --cov-fail-under=50 \
    -v \
    "$@"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "\n${GREEN}All tests passed!${NC}"
    echo -e "Coverage report: coverage_html/index.html"
else
    echo -e "\n${RED}Tests failed!${NC}"
fi

exit $EXIT_CODE
