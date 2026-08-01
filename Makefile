.PHONY: all build audit test pdf clean

export PYTHONPATH := $(CURDIR):$(CURDIR)/target/release

all: build test audit pdf

build:
	python shbt_simulate.py --build

test:
	pytest tests/

audit:
	python examples/run_audit.py

pdf:
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex

clean:
	rm -f *.aux *.log *.out *.toc *.synctex.gz sim_results.tex
