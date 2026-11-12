# `data`

Canonical in-repo input datasets. MNIST lives only at `data/MNIST/` (torchvision layout: `raw/` idx files). The shipped train split is a **100-image random subset** in the same idx format (see `subset_manifest.json`); rebuild with `python tools/build_mnist_subset.py --n 100`. Do not keep a second copy under `examples/data/`. Experimental TIFF archives are NOT stored here.

Created/retained by Task 02 architecture setup. Do not write paper artifacts here; generators under `tools/` write regenerable products to the local gitignored `evidence/` tree.
