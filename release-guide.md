new release checklist
=====================

* bump version in setup.py
* bump version in sphinx
* create github release with tag
* python setup.py sdist bdist_wheel
* twine upload --skip-existing dist/*
