# the code snippet is based on https://packaging.python.org/en/latest/tutorials/packaging-projects/
# be careful for releasing to official PyPI
# as this versioning is non-reversible
python3 -m pip install --upgrade build
python3 -m build
python3 -m pip install --upgrade twine
python3 -m twine upload dist/*
echo "Done!"
