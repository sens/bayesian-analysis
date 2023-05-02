%.html: %.ipynb
	jupytext --to Rmd $*.ipynb
	mv $*.Rmd $*.jmd
	julia --threads 16 -e 'using Weave; weave("$*.jmd",doctype="pandoc2html",pandoc_options = ["--toc", "--toc-depth= 3"])'
