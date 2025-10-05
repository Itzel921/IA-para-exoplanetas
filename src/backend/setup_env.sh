#!/bin/bash

# =============================================================================
# Virtual Environment Setup Script for Exoplanet Detection Backend
# NASA Space Apps Challenge 2025
# =============================================================================

set -e  # Exit on any error

# Configuration
VENV_NAME="exoplanet_backend_env"
PYTHON_VERSION="3.8"  # Minimum required version
PROJECT_DIR=$(dirname "$(readlink -f "$0")")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python_version() {
    local python_cmd=$1
    if command_exists "$python_cmd"; then
        local version=$($python_cmd --version 2>&1 | cut -d' ' -f2)
        local major=$(echo $version | cut -d'.' -f1)
        local minor=$(echo $version | cut -d'.' -f2)
        
        if [[ $major -eq 3 && $minor -ge 8 ]]; then
            echo "$python_cmd"
            return 0
        fi
    fi
    return 1
}

# Function to find suitable Python interpreter
find_python() {
    log_info "Searching for Python interpreter (>= 3.8)..."
    
    # Try different Python commands
    for cmd in python3.11 python3.10 python3.9 python3.8 python3 python; do
        if python_version=$(check_python_version "$cmd"); then
            log_success "Found suitable Python: $python_version ($($python_version --version))"
            echo "$python_version"
            return 0
        fi
    done
    
    log_error "No suitable Python interpreter found (requires Python >= 3.8)"
    log_info "Please install Python 3.8 or higher and try again"
    return 1
}

# Main setup function
setup_environment() {
    log_info "=== Exoplanet Detection Backend - Environment Setup ==="
    
    # Change to project directory
    cd "$PROJECT_DIR"
    log_info "Working directory: $(pwd)"
    
    # Find Python interpreter
    if ! PYTHON_CMD=$(find_python); then
        exit 1
    fi
    
    # Remove existing virtual environment if it exists
    if [[ -d "$VENV_NAME" ]]; then
        log_warning "Existing virtual environment found. Removing..."
        rm -rf "$VENV_NAME"
    fi
    
    # Create virtual environment
    log_info "Creating virtual environment: $VENV_NAME"
    $PYTHON_CMD -m venv "$VENV_NAME"
    
    # Activate virtual environment
    log_info "Activating virtual environment..."
    source "$VENV_NAME/bin/activate"
    
    # Upgrade pip
    log_info "Upgrading pip..."
    pip install --upgrade pip
    
    # Install wheel for faster package installations
    log_info "Installing wheel..."
    pip install wheel
    
    # Install requirements
    if [[ -f "requirements.txt" ]]; then
        log_info "Installing Python dependencies from requirements.txt..."
        pip install -r requirements.txt
    elif [[ -f "../../requirements.txt" ]]; then
        log_info "Installing Python dependencies from ../../requirements.txt..."
        pip install -r ../../requirements.txt
    else
        log_warning "No requirements.txt found. Installing basic dependencies..."
        pip install fastapi uvicorn pandas numpy scikit-learn
    fi
    
    # Create .env file if it doesn't exist
    if [[ ! -f ".env" && -f ".env.template" ]]; then
        log_info "Creating .env file from template..."
        cp .env.template .env
        log_info "Please review and customize .env file as needed"
    fi
    
    # Create necessary directories
    log_info "Creating project directories..."
    mkdir -p logs models data temp tests
    
    # Generate activation script
    log_info "Creating activation script..."
    cat > activate_env.sh << 'EOF'
#!/bin/bash
# Activation script for Exoplanet Detection Backend

PROJECT_DIR=$(dirname "$(readlink -f "$0")")
cd "$PROJECT_DIR"

if [[ -d "exoplanet_backend_env" ]]; then
    echo "Activating virtual environment..."
    source exoplanet_backend_env/bin/activate
    echo "Virtual environment activated. Project directory: $(pwd)"
    echo "To deactivate, run: deactivate"
else
    echo "Virtual environment not found. Run setup_env.sh first."
    exit 1
fi
EOF
    chmod +x activate_env.sh
    
    # Generate Windows activation script
    log_info "Creating Windows activation script..."
    cat > activate_env.bat << 'EOF'
@echo off
rem Activation script for Exoplanet Detection Backend (Windows)

cd /d "%~dp0"

if exist "exoplanet_backend_env" (
    echo Activating virtual environment...
    call exoplanet_backend_env\Scripts\activate.bat
    echo Virtual environment activated. Project directory: %CD%
    echo To deactivate, run: deactivate
) else (
    echo Virtual environment not found. Run setup_env.bat first.
    pause
    exit /b 1
)
EOF
    
    log_success "=== Environment Setup Complete ==="
    echo
    log_info "Virtual environment created: $VENV_NAME"
    log_info "Python version: $($PYTHON_CMD --version)"
    log_info "Pip version: $(pip --version)"
    echo
    log_info "To activate the environment:"
    log_info "  Linux/Mac: source activate_env.sh"
    log_info "  Windows:   activate_env.bat"
    log_info "  Manual:    source $VENV_NAME/bin/activate"
    echo
    log_info "To run the backend:"
    log_info "  python main.py"
    log_info "  or: uvicorn main:app --reload"
    echo
    log_info "To deactivate: deactivate"
}

# Function to check installation
check_installation() {
    log_info "Verifying installation..."
    
    # Check if virtual environment exists
    if [[ ! -d "$VENV_NAME" ]]; then
        log_error "Virtual environment not found"
        return 1
    fi
    
    # Activate and check packages
    source "$VENV_NAME/bin/activate"
    
    # Check key packages
    local required_packages=("fastapi" "uvicorn" "pandas" "numpy" "scikit-learn")
    local missing_packages=()
    
    for package in "${required_packages[@]}"; do
        if ! pip show "$package" > /dev/null 2>&1; then
            missing_packages+=("$package")
        fi
    done
    
    if [[ ${#missing_packages[@]} -eq 0 ]]; then
        log_success "All required packages are installed"
        return 0
    else
        log_error "Missing packages: ${missing_packages[*]}"
        return 1
    fi
}

# Function to clean environment
clean_environment() {
    log_info "Cleaning environment..."
    if [[ -d "$VENV_NAME" ]]; then
        rm -rf "$VENV_NAME"
        log_success "Virtual environment removed"
    fi
    
    # Remove generated scripts
    rm -f activate_env.sh activate_env.bat
    log_success "Generated scripts removed"
}

# Main script logic
case "${1:-setup}" in
    "setup")
        setup_environment
        ;;
    "check")
        check_installation
        ;;
    "clean")
        clean_environment
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [command]"
        echo
        echo "Commands:"
        echo "  setup (default) - Set up virtual environment and install dependencies"
        echo "  check          - Verify installation"
        echo "  clean          - Remove virtual environment"
        echo "  help           - Show this help message"
        ;;
    *)
        log_error "Unknown command: $1"
        log_info "Run '$0 help' for usage information"
        exit 1
        ;;
esac