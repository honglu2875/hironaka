SHELL=/bin/bash
LINT_PATHS=hironaka/ test/ setup.py

type:
	pytype -j auto

lint:
	flake8 ${LINT_PATHS} --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 ${LINT_PATHS} --count --exit-zero --statistics

format:
	# Sort imports
	isort ${LINT_PATHS}
	# Reformat using black
	black -l 127 ${LINT_PATHS}

check-codestyle:
	# Sort imports
	isort --check ${LINT_PATHS}
	# Reformat using black
	black --check -l 127 ${LINT_PATHS}

commit-checks: format type lint

doc:
	cd docs && make html

spelling:
	cd docs && make spelling

clean:
	cd docs && make clean
