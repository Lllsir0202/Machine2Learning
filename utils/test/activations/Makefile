.PHONY: test clean

TEST_UNITS_CMD = python3 -m unittest discover ./

all: test

test:
	@echo "Running all unit tests..."
	$(TEST_UNITS_CMD)

clean:
	@echo "Cleaning up..."
	-find . -type f -name "*.pyc" -delete
	-find . -type d -name "__pycache__" -delete