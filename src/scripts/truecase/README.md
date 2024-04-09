# REL

## Truecasing

Flair does not tag lowercase input. The best solution is to train a better FLAIR tagger, but if you are in a hurry, _truecasing_ the input can be an alternative.

This example uses [daltonfury42/truecase](https://github.com/daltonfury42/truecase/).

Generic way to set `PYTHONPATH`:

	export PYTHONPATH=$(python -c "import site, os, platform; print(os.path.join(site.USER_BASE, 'lib', 'python' + str.join('.',platform.python_version_tuple()[0:2]), 'site-packages'))"):$PYTHONPATH

Install `nltk`:

    pip install --user -U nltk

Install `truecase`:

    git clone git@github.com:daltonfury42/truecase.git
    cd truecase/
    pip install --user -e .

Truecase the queries in `allqueries.txt`:

    python truecase-m.py
	
Tag the queries in `allQueries.txt` (output from above):

    python relq.py allQueries.txt > allQtagged.txt
	
	
