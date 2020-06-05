#!/usr/bin/env sh

REQUIREMENTS_FILE=ci_torch_cpu_requirements.txt
python3 gen_torch_cpu_requirements.py --file $REQUIREMENTS_FILE
pip3 install -r $REQUIREMENTS_FILE

pip3 install -r requirements-test.txt

pip3 install --upgrade .  codecov

pip3 list
