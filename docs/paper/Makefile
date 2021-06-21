ROOT=dreamsound
LTX=pdflatex
BIB=biber
PDC=pandoc
OUTDIR=out
LTXDIR=src

PDCSTY=~/bin/style/pandoc.css
PDCCSL=~/bin/style/author-date.csl
LTXFLAGS=-interaction=nonstopmode -file-line-error
LTXFLAGS+=-output-format=pdf -jobname=$(ROOT) -output-directory=$(OUTDIR)
BIBFLAGS=--output-directory $(OUTDIR) --input-directory $(OUTDIR) 
PDCFLAGS=-f latex -s --wrap none --katex 
PDCFLAGS+=--toc --toc-depth 4 --css $(PDCSTY) 
PDCFLAGS+=--metadata pagetitle=$(ROOT) 
PDCFLAGS+=-C --bibliography $(ROOT).bib --csl $(PDCCSL)

SRC=$(ROOT).tex

all: 
	make paper
	make pandoc

paper: $(LTXDIR)/$(SRC)
	cd $(LTXDIR) && $(LTX) $(LTXFLAGS) $(SRC)
	cd $(LTXDIR) && $(BIB) $(BIBFLAGS) $(ROOT)
	cd $(LTXDIR) && $(LTX) $(LTXFLAGS) $(SRC)
	cd $(LTXDIR) && $(LTX) $(LTXFLAGS) $(SRC)
	
pandoc: $(LTXDIR)/$(SRC)
	cd $(LTXDIR) && $(PDC) $(PDCFLAGS) -t html -o $(OUTDIR)/$(ROOT).html $(SRC)
	cd $(LTXDIR) && $(PDC) $(PDCFLAGS) -t docx -o $(OUTDIR)/$(ROOT).docx $(SRC)
	cd $(LTXDIR) && $(PDC) $(PDCFLAGS) -t gfm  -o $(OUTDIR)/$(ROOT).md   $(SRC)

clean:
# 	cd $(LTXDIR) && rm $(ROOT).{bcf,aux,bbl,blg,log,out,run.xml}
	echo rm $(LTXDIR)/$(OUTDIR)/*

.PHONY: all clean paper