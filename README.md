# Satellite re-entry simulator and predictor

## Setup

This package requires `Python>=3.10`.

For development purposes, you can install this package - after cloning to a local directory and navigating into the repository - with the command

```
python -m pip install -e .[dev]
```

This will install the package in editable mode, allowing the package to be modified and changes to be applied immediately in the local environment.

An end user might use the following command to install this package into their current environment:

```
python -m pip install git+https://github.com/ES98B-Mir-project-23/mir-orbiter.git@main#egg=mir-satellite-deorbiter
```

After either of the above, the package will be available as `deorbit` in the python environment.

For usage examples see /examples/
