# Satellite re-entry simulator and predictor

## Setup

This package requires `Python>=3.10`.

For development purposes, you can clone the repository to a local directory with

```
git clone git@github.com:ES98B-Mir-project-23/mir-orbiter.git
cd mir-orbiter
```
and install will the pip command
```
python -m pip install -e .[dev]
```

This will install the package in editable mode, allowing the package to be modified and changes to be applied immediately in the local environment.

An end user might use the following command to install this package into their current environment directly from the github repo:

```
python -m pip install mir-satellite-deorbiter@git+https://github.com/ES98B-Mir-project-23/mir-orbiter.git@main
```

After either of the above, the package will be available as `deorbit` in the python environment.

## Examples

For usage examples see [examples/](examples)

## API documentation

To build the documentation, you must first have installed the package's dev dependencies with

```
python -m pip install mir-satellite-deorbiter[dev]@git+https://github.com/ES98B-Mir-project-23/mir-orbiter.git@main
```

The following will build the documentation in [docs/build](docs/build/):

```
sphinx-build -M html docs/source/ docs/build/
```
