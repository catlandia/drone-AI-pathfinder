#!/bin/bash
#
# Drone AI Path Finder - Easy Installer
#
# Usage:
#   ./install.sh          # Basic install
#   ./install.sh --full   # Install with AI training support
#   ./install.sh --help   # Show help
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}          ${GREEN}Drone AI Path Finder - Installer${NC}                 ${BLUE}║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_step() {
    echo -e "${GREEN}▶${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

show_help() {
    echo "Drone AI Path Finder - Installer"
    echo ""
    echo "Usage: ./install.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --basic     Install core package only (default)"
    echo "  --viz       Install with visualization (matplotlib, pygame)"
    echo "  --full      Install everything including AI training (stable-baselines3)"
    echo "  --venv      Create virtual environment first"
    echo "  --help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./install.sh                  # Quick install"
    echo "  ./install.sh --full --venv    # Full install in virtual environment"
    echo ""
}

# Parse arguments
INSTALL_TYPE="basic"
USE_VENV=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --basic)
            INSTALL_TYPE="basic"
            shift
            ;;
        --viz)
            INSTALL_TYPE="viz"
            shift
            ;;
        --full)
            INSTALL_TYPE="full"
            shift
            ;;
        --venv)
            USE_VENV=true
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

print_header

# Check Python
print_step "Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON=python3
    PIP=pip3
elif command -v python &> /dev/null; then
    PYTHON=python
    PIP=pip
else
    print_error "Python not found! Please install Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$($PYTHON --version 2>&1 | awk '{print $2}')
print_success "Found Python $PYTHON_VERSION"

# Create virtual environment if requested
if [ "$USE_VENV" = true ]; then
    print_step "Creating virtual environment..."
    $PYTHON -m venv venv
    source venv/bin/activate
    print_success "Virtual environment created and activated"
    echo ""
    print_warning "To activate later, run: source venv/bin/activate"
    echo ""
fi

# Upgrade pip
print_step "Upgrading pip..."
$PIP install --upgrade pip -q

# Install package
print_step "Installing drone-ai-pathfinder..."

case $INSTALL_TYPE in
    basic)
        $PIP install -e . -q
        print_success "Core package installed"
        ;;
    viz)
        $PIP install -e ".[viz]" -q
        print_success "Package installed with visualization"
        ;;
    full)
        $PIP install -e ".[all]" -q
        print_step "Installing stable-baselines3 for AI training..."
        $PIP install stable-baselines3 -q
        print_success "Full package installed with AI training support"
        ;;
esac

# Verify installation
print_step "Verifying installation..."
$PYTHON -c "from drone_ai import DroneEnv, PathPlanner; print('Import OK')" 2>/dev/null && \
    print_success "Installation verified!" || \
    print_error "Installation verification failed"

# Print next steps
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Installation complete!${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Quick start commands:"
echo ""
echo -e "  ${YELLOW}# Run demo (math-based controller)${NC}"
echo "  python -m drone_ai.demo --task delivery_route"
echo ""
echo -e "  ${YELLOW}# Run with visualization${NC}"
echo "  python -m drone_ai.demo --task delivery_route --render"
echo ""

if [ "$INSTALL_TYPE" = "full" ]; then
echo -e "  ${YELLOW}# Train AI with reinforcement learning${NC}"
echo "  python -m drone_ai.train_rl --algo ppo --task hover"
echo ""
fi

echo -e "  ${YELLOW}# Run curriculum learning${NC}"
echo "  python -m drone_ai.learning_sequence"
echo ""
echo "For more info, see README.md"
echo ""
