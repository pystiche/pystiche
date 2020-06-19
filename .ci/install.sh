#!/usr/bin/env sh

pip3 install pytorch_wheel_installer
pwi --pip-cmd "pip3 install"

pip3 install -r requirements-test.txt

pip3 install --upgrade .  codecov

pip3 list
