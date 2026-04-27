# bonus/Makefile
# Builds lda_cuda.so shared library with CUDA support
#
# Usage:
#   make              — auto-detect GPU arch
#   make ARCH=sm_86   — force specific arch (RTX 30xx = sm_86, RTX 40xx = sm_89)
#
# Common arch values:
#   sm_61  →  GTX 10xx (Pascal)
#   sm_75  →  RTX 20xx / GTX 16xx (Turing)
#   sm_80  →  A100 (Ampere)
#   sm_86  →  RTX 30xx (Ampere)
#   sm_89  →  RTX 40xx (Ada Lovelace)

NVCC     = nvcc
TARGET   = lda_cuda.so
SRC      = lda_cuda.cu

# Auto-detect GPU architecture if not specified
ifndef ARCH
    ARCH := $(shell python3 -c "\
import subprocess, re, sys; \
out = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'], \
    capture_output=True, text=True).stdout.strip(); \
cap = out.split('\n')[0].strip().replace('.',''); \
print('sm_' + cap) if cap else print('sm_75')" 2>/dev/null || echo sm_75)
endif

NVCCFLAGS = -O3 -arch=$(ARCH) --compiler-options -fPIC,-O3 -std=c++14
LDFLAGS   = -shared -lm

.PHONY: all clean info

all: info $(TARGET)

info:
	@echo "Building for architecture: $(ARCH)"
	@nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null || true

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ $<
	@echo "  ✓  $(TARGET) built"

clean:
	rm -f $(TARGET)
