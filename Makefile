JUPYTEXT ?= uv tool run jupytext

NOTEBOOK_SOURCES = lda_hpc.py lda_cuda_comparison.py
NOTEBOOKS = $(NOTEBOOK_SOURCES:.py=.ipynb)

.PHONY: all notebooks serial parallel bonus clean

all: serial parallel notebooks

notebooks: $(NOTEBOOKS)

%.ipynb: %.py
	$(JUPYTEXT) --to notebook $<

serial:
	$(MAKE) -C serial

parallel:
	$(MAKE) -C parallel

bonus:
	$(MAKE) -C bonus

clean:
	rm -f $(NOTEBOOKS)
	$(MAKE) -C serial clean
	$(MAKE) -C parallel clean
	$(MAKE) -C bonus clean
