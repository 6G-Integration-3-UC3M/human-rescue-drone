VENV_PATH := venv

.PHONY: all
all: run

.PHONY: run
run:
	@echo "Running the drone..."
	@source $(VENV_PATH)/bin/activate && python main.py

.PHONY: install
install:
	@echo "Installing the drone..."
	@if [ -d $(VENV_PATH) ]; then \
		echo "Virtual environment already exists"; \
	else \
		echo "Creating virtual environment at $(VENV_PATH)"; \
		python3 -m venv $(VENV_PATH); \
	fi
	@echo "Activating the virtual environment and installing dependencies..."
	@source $(VENV_PATH)/bin/activate && pip install -r requirements.txt

.PHONY: clean
clean:
	@echo "Cleaning up..."
	@rm -rf $(VENV_PATH)

.PHONY: test
test:
	@echo "Running tests..."
	@source $(VENV_PATH)/bin/activate && python -m pytest tests
