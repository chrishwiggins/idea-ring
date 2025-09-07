.PHONY: all run run-optimized install test clean generate-requirements help

python = /usr/bin/python3

# Default target: install dependencies and run optimized version
all: install run-optimized

# Basic run with closure-constrained TSP
run: install
	@echo "Running Idea Ring with basic optimization..."
	$(python) idea-ring.py

# Advanced run with 2-opt optimization
run-optimized: install
	@echo "Running Idea Ring with advanced 2-opt optimization..."
	$(python) idea-ring.py --optimize

# Run with custom file
run-file: install
	@echo "Usage: make run-file FILE=your_file.txt"
	$(python) idea-ring.py -f $(FILE)

# Install required Python packages
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt

# Test the installation and basic functionality
test: install
	@echo "Testing idea ring functionality..."
	@echo "This will process the default datafile.txt:"
	$(python) idea-ring.py | head -10
	@echo "Test completed successfully!"

# Clean up any temporary files
clean:
	@echo "Cleaning up temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# Generate/update requirements.txt from current environment
generate-requirements:
	@echo "Generating requirements.txt from current environment..."
	pip freeze > requirements.txt

# Show help
help:
	@echo "Idea Ring - Transform brainstormed ideas into optimized timelines"
	@echo ""
	@echo "Available targets:"
	@echo "  all              - Install dependencies and run optimized version (default)"
	@echo "  run              - Run with basic TSP optimization"
	@echo "  run-optimized    - Run with advanced 2-opt optimization"
	@echo "  run-file FILE=x  - Run with custom input file"
	@echo "  install          - Install Python dependencies"
	@echo "  test             - Test basic functionality"
	@echo "  clean            - Remove temporary files"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make                          # Install and run optimized version"
	@echo "  make run-optimized           # Use advanced optimization"
	@echo "  make run-file FILE=my.txt    # Process custom file"
