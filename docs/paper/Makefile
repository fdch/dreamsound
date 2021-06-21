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
BIBFLAGS=--output-directory $(OUTDIR) --input-directory .
PDCFLAGS=-f latex -s  --wrap=preserve
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
	pandoc --toc --toc-depth 4 --css $(PDCSTY) -s --wrap=preserve -f latex -t html -C --bibliography src/dreamsound.bib --csl $(PDCCSL) -o src/out/dreamsound.html src/dreamsound.tex 
	pandoc -s --wrap=preserve -f html -t gfm -o src/out/dreamsound.md src/out/dreamsound.html 
	pandoc -s --wrap=preserve -f html -t docx -o src/out/dreamsound.docx src/out/dreamsound.html 
	pandoc --wrap=preserve -f html -t plain -o src/out/dreamsound.txt src/out/dreamsound.html 

clean:
# 	cd $(LTXDIR) && rm $(ROOT).{bcf,aux,bbl,blg,log,out,run.xml}
	echo rm $(LTXDIR)/$(OUTDIR)/*

.PHONY: all clean paper