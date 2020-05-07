#!/usr/bin/env sh

coverage run --rcfile=.coveragerc -m pytest -c pytest.ini --skip-large-download
