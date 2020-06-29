# Install with: $ python3 setup.py install

# based on: https://stackoverflow.com/questions/49031491/import-from-my-package-recognized-by-pycharm-but-not-by-command-line


from setuptools import setup, find_packages

setup(
	name='rl-discount',
	packages=find_packages(),
	version='0.0.1',
)


