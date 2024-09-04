.PHONY: run install generate-requirements

python = /usr/bin/python3  # Ensure this is correct

run: install
	$(python) idea-ring.py

install:
	pip install -r requirements.txt

generate-requirements:
	pip freeze > requirements.txt
