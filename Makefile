NVCC ?= nvcc
CXXFLAGS := -O2 -std=c++14
NVCCFLAGS := $(CXXFLAGS) -Xcompiler -Wall

SRC := src/box_filter.cu
BIN := bin/box_filter

.PHONY: all clean run

all: $(BIN)

$(BIN): $(SRC)
	mkdir -p bin data/output
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run: $(BIN)
	./run.sh

clean:
	rm -f $(BIN)
	rm -rf data/output/*
