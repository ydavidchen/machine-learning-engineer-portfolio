# Exercise: Publishing a Simple _PyPI_ Package

## Procedure

```shell
cd custom-distribution;
python3 setup.py sdist;
pip3 install twine

## Upload to the PyPI Test repo
twine upload --repository-url https://test.pypi.org/legacy/ dist/*
pip3 install --index-url https://test.pypi.org/simple/ custom-distributions

## Upload to the PyPI
twine upload dist/*;
pip3 install custom-distributions;
```
