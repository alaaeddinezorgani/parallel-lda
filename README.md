# LDA + Parallel Scatter Matrices — Topic 42

Fisher's LDA from scratch with OpenMP-parallelised scatter matrix computation.  
Dataset: HAR Smartphones (10 299 × 561 features, 6 classes).

---

## Repository layout

```
.
├── serial/
│   ├── lda_serial.c        # Serial implementation (no parallel constructs)
│   └── Makefile
├── parallel/
│   ├── lda_openmp.c        # OpenMP implementation (across-sample parallelism)
│   └── Makefile
├── bonus/                  # GPU implementation (CUDA) — optional +15%
├── data/
│   └── download.sh         # Kaggle dataset download script
├── lda_hpc.ipynb           # Jupyter notebook — orchestrator & visualisation
└── README.md
```

---

## Requirements

```bash
pip install numpy pandas scipy scikit-learn matplotlib jupyter
```

GCC with OpenMP support (default on Linux; on macOS use Homebrew gcc, not clang):

```bash
# macOS
brew install gcc
export CC=gcc-13   # or whatever version brew installed
```

---

## Compile

```bash
# Serial shared library
make -C serial

# Parallel shared library (OpenMP)
make -C parallel
```

Both produce `.so` files loaded by the notebook via ctypes.

---

## Dataset download

```bash
# Requires Kaggle API credentials (~/.kaggle/kaggle.json)
bash data/download.sh
```

Or manually download from:  
https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones  
and place `train.csv` and `test.csv` in the `data/` folder.

---

## Run

```bash
jupyter notebook lda_hpc.ipynb
```

Execute all cells in order. The notebook will:
1. Compile both C libraries
2. Load and inspect the dataset
3. Run serial LDA + measure S_W time
4. Sweep parallel LDA across thread counts {1, 2, 4, 6, 8}
5. Verify correctness (serial ≈ parallel to machine precision)
6. Plot speedup, efficiency, and Amdahl curves
7. Project data onto LDA axes and visualise
8. Report classification accuracy (1-NN on projected space)

---

## Parallelisation strategy

**Option A — across samples (chosen)**

The within-class scatter matrix accumulation:

```
S_W = Σ_i (x_i − μ_{c(i)}) (x_i − μ_{c(i)})^T
```

is decomposed by sample index. Each OpenMP thread owns a **private** `S_W` buffer,
processes its chunk of rows, then all private buffers are reduced (summed) serially.
This avoids all race conditions and false sharing on the matrix itself.

**Why not across classes?**  
6 classes → hard cap of 6× speedup. Option A scales with `n` (10 K rows),
giving meaningful speedup at 8+ threads.

---

## AI tool declaration

GitHub Copilot / Claude were used only for boilerplate scaffolding (ctypes bindings,
Makefile syntax). All algorithmic code (scatter matrix loops, OpenMP decomposition,
eigen-solve pipeline) was written and understood by the team.
