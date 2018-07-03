DOCUMENTS := $(patsubst %.tex, %.pdf, $(wildcard *.tex))

.PHONY: all

all: $(DOCUMENTS)

%.pdf: %.tex
	pdflatex $<
