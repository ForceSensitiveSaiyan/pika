#!/bin/bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MODEL="${OLLAMA_MODEL:-mistral:7b}"
PIKA_PORT="${PIKA_PORT:-8000}"

echo -e "${BLUE}"
echo "╔═══════════════════════════════════════╗"
echo "║           PIKA Setup Script           ║"
echo "║   Self-hosted RAG for Small Business  ║"
echo "╚═══════════════════════════════════════╝"
echo -e "${NC}"

# Check for Docker
echo -e "${YELLOW}[1/5] Checking prerequisites...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed.${NC}"
    echo "Please install Docker from https://docs.docker.com/get-docker/"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo -e "${RED}Error: Docker daemon is not running.${NC}"
    echo "Please start Docker and try again."
    exit 1
fi

if ! command -v docker compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}Error: Docker Compose is not available.${NC}"
    echo "Please install Docker Compose or update Docker."
    exit 1
fi

echo -e "${GREEN}✓ Docker is ready${NC}"

# Create documents directory if it doesn't exist
echo -e "${YELLOW}[2/5] Setting up directories...${NC}"
mkdir -p documents
echo -e "${GREEN}✓ Documents directory ready${NC}"

# Start services
echo -e "${YELLOW}[3/5] Starting services...${NC}"
docker compose up -d --build

# Wait for Ollama to be healthy
echo -e "${YELLOW}[4/5] Waiting for Ollama to be ready...${NC}"
MAX_ATTEMPTS=30
ATTEMPT=0
while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    if docker compose exec -T ollama curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Ollama is ready${NC}"
        break
    fi
    ATTEMPT=$((ATTEMPT + 1))
    echo "  Waiting for Ollama... (attempt $ATTEMPT/$MAX_ATTEMPTS)"
    sleep 2
done

if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
    echo -e "${RED}Error: Ollama failed to start${NC}"
    docker compose logs ollama
    exit 1
fi

# Pull the model
echo -e "${YELLOW}[5/5] Pulling AI model ($MODEL)...${NC}"
echo "  This may take a few minutes on first run..."
docker compose exec -T ollama ollama pull $MODEL

# Wait for PIKA to be healthy
echo -e "${YELLOW}Waiting for PIKA to be ready...${NC}"
ATTEMPT=0
while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    if curl -sf http://localhost:$PIKA_PORT/api/v1/health > /dev/null 2>&1; then
        break
    fi
    ATTEMPT=$((ATTEMPT + 1))
    sleep 2
done

# Success message
echo ""
echo -e "${GREEN}╔═══════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         PIKA is ready! 🎉             ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${BLUE}Web Interface:${NC}  http://localhost:$PIKA_PORT"
echo -e "  ${BLUE}API Docs:${NC}       http://localhost:$PIKA_PORT/docs"
echo ""
echo -e "  ${YELLOW}Quick Start:${NC}"
echo "  1. Drop documents into the ./documents folder"
echo "  2. Click 'Refresh Index' in the web interface"
echo "  3. Start asking questions!"
echo ""
echo -e "  ${YELLOW}Commands:${NC}"
echo "  Stop:    docker compose down"
echo "  Logs:    docker compose logs -f"
echo "  Restart: docker compose restart"
echo ""
