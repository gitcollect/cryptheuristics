# cryptoheuristics
Heuristic Methods of Cryptanalysis repo

## TODO
* test kasiski_examination and add examples
* improve documentation in evaluators module
* repair selection methods to work with new `Solution` classes

## Docs compilation

    cd doc
    sphinx-quickstart
    sphinx-apidoc -o source ../src/cryptheuristics/
    make html

It may be necessary to install read the docs template first:

    pip3 install sphinx_rtd_theme
