# Makefile
# 
# Converts Markdown to other formats (HTML, PDF, DOCX, RTF, ODT, EPUB) using Pandoc
# <http://johnmacfarlane.net/pandoc/>
#
# Run "make" (or "make all") to convert to all other formats
#
# Run "make clean" to delete converted files

# Convert all files in this directory that have a .md suffix
SOURCE_DIR=src
OUTPUT_DIR=pdf
SOURCE_DOCS := $(wildcard $(SOURCE_DIR)/*.md $(SOURCE_DIR)/*/*.md)

EXPORTED_DOCS=\
 $(SOURCE_DOCS:$(SOURCE_DIR)/%.md=$(OUTPUT_DIR)/%.pdf)
#  $(SOURCE_DOCS:.md=.html) \
#  $(SOURCE_DOCS:.md=.pdf) \
#  $(SOURCE_DOCS:.md=.docx) \
#  $(SOURCE_DOCS:.md=.rtf) \
#  $(SOURCE_DOCS:.md=.odt) \
#  $(SOURCE_DOCS:.md=.epub)

RM=rm
MKDIR=mkdir

PANDOC=pandoc
PANDOC_OPTIONS=
PANDOC_HTML_OPTIONS=--to html5
PANDOC_PDF_OPTIONS=--pdf-engine=xelatex
PANDOC_DOCX_OPTIONS=
PANDOC_RTF_OPTIONS=
PANDOC_ODT_OPTIONS=
PANDOC_EPUB_OPTIONS=--to epub3


# Pattern-matching Rules

$(OUTPUT_DIR)/%.html : $(SOURCE_DIR)/%.md
	$(MKDIR) -p $(dir $@)
	$(PANDOC) $(PANDOC_OPTIONS) $(PANDOC_HTML_OPTIONS) -o $@ $<

$(OUTPUT_DIR)/%.pdf : $(SOURCE_DIR)/%.md
	$(MKDIR) -p $(dir $@)
	$(PANDOC) $(PANDOC_OPTIONS) $(PANDOC_PDF_OPTIONS) -o $@ $<
	
$(OUTPUT_DIR)/%.docx : $(SOURCE_DIR)/%.md
	$(MKDIR) -p $(dir $@)
	$(PANDOC) $(PANDOC_OPTIONS) $(PANDOC_DOCX_OPTIONS) -o $@ $<

$(OUTPUT_DIR)/%.rtf : $(SOURCE_DIR)/%.md
	$(MKDIR) -p $(dir $@)
	$(PANDOC) $(PANDOC_OPTIONS) $(PANDOC_RTF_OPTIONS) -o $@ $<

$(OUTPUT_DIR)/%.odt : $(SOURCE_DIR)/%.md
	$(MKDIR) -p $(dir $@)
	$(PANDOC) $(PANDOC_OPTIONS) $(PANDOC_ODT_OPTIONS) -o $@ $<

$(OUTPUT_DIR)/%.epub : $(SOURCE_DIR)/%.md
	$(MKDIR) -p $(dir $@)
	$(PANDOC) $(PANDOC_OPTIONS) $(PANDOC_EPUB_OPTIONS) -o $@ $<


# Targets and dependencies

.PHONY: all clean

all : $(EXPORTED_DOCS)

clean:
	$(RM) $(EXPORTED_DOCS)

test:
	@echo $(SOURCE_DOCS)
	@echo $(EXPORTED_DOCS)