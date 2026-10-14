# DeepCoEx

**Beyond Co-Localization: DeepCoEx Learns Exclusion-Aware Spatial Representations in Mass Spectrometry Imaging via Dual-Contrastive Learning**

[Analytical Chemistry paper](https://doi.org/10.1021/acs.analchem.6c02338)

DeepCoEx is a self-supervised representation-learning workflow for mass spectrometry imaging (MSI). It learns ion-image embeddings that support the exploration of both spatial co-localization and candidate complementary (exclusion-like) spatial associations. The released scripts cover tissue-mask preparation, data augmentation, dual-contrastive training, feature extraction, and dimensionality reduction.

## Method overview

For each ion image, DeepCoEx constructs two **positive-associated views (PAVs)** using stochastic image augmentations. A **negative-associated view (NAV)** is obtained by applying tissue-mask-based intensity inversion to one of the PAVs. A ResNet-18 encoder and projection/prediction heads are trained with a dual-contrastive objective that encourages agreement between the PAVs and separation between the PAVs and NAV. The trained ResNet-18 is then used to extract 512-dimensional ion-image features, which are projected into a 20-dimensional UMAP space.

The NAV is a *counterfactual training view*, not a biologically validated negative ion pair. Likewise, negative-association candidates inferred from learned representations should not be interpreted as proof of biochemical mutual exclusion.

## Repository files

| File | Description |
| --- | --- |
| `run.py` | Command-line entry point for training, feature extraction, and UMAP reduction. |
| `DeepION.py` | Training, 512-dimensional feature extraction, and UMAP functions. `DeepION` is a legacy filename/function prefix retained in this implementation. |
| `byol_pytorch.py` | Encoder wrapper, projection/prediction heads, target-encoder updates, and dual-contrastive loss. |
| `Augmentation.py` | Custom intensity-dependent missing and random-missing augmentations; imports additional augmentations from Kornia. |
| `neg_augment.py` | Tissue-mask-based intensity-inversion augmentation for the NAV. |
| `DownStreamTask.py` | Dataset-specific utility for generating a tissue mask and saving the tensor used by `neg_augment.py`. Despite its filename, this script does not implement ion retrieval, clustering, or molecular-network construction. |

## Requirements

- Python with a CUDA-enabled PyTorch installation
- `torch`, `torchvision`
- `kornia`
- `numpy`
- `scikit-learn`
- `umap-learn`
- `matplotlib` (used by the mask-preparation utility)

An example installation command, after installing an appropriate CUDA-compatible PyTorch and torchvision build, is:

```bash
pip install numpy scikit-learn umap-learn kornia matplotlib
```

Specific package versions are not pinned in these scripts; use mutually compatible versions of PyTorch, torchvision, and Kornia.

## Input data

### MSI matrix

The training entry point expects a preprocessed NumPy `.npy` matrix with shape:

```text
(number_of_ions, height * width)
```

Each row represents one flattened ion image. Supply the spatial dimensions as `--input_shape HEIGHT WIDTH`; their product must equal the number of matrix columns. The implementation transposes this matrix internally, normalizes each ion-image column by its total intensity, and then applies min–max scaling. Ion images with zero total intensity should be removed or handled before running the script to avoid division by zero.

### Tissue-mask tensor

The negative augmentation reads a tissue-mask tensor from `--tensor_path`. The supplied `DownStreamTask.py` creates a binary tissue mask (tissue = 1, background = 0), expands it to `(100, 3, HEIGHT, WIDTH)`, and saves it with `torch.save()`.

**Before running `DownStreamTask.py`, edit its dataset path, image dimensions, and output filenames to match your data.** This utility currently assumes its input array is organized as `(height * width, number_of_ions)` when computing the mask. This is the transpose of the matrix layout expected by `run.py`; adapt the input or script accordingly.

```bash
python DownStreamTask.py
```

The default output tensor filename in the supplied utility is `tensor-pg10-3-train.pth`.

## Run the workflow

Place the six Python scripts together in the repository directory.

A command-line example is:

```bash
python run.py \
  --input_Matrix data/gz-filter.npy \
  --input_PeakList data/gz-peak_list.csv \
  --input_shape 110 236 \
  --ion_mode negative \
  --tensor_path data/gz-train.pth \
  --mode COL
```

Replace all example paths and dimensions with those of your dataset. `--input_PeakList` and `--ion_mode` are required by the current command-line parser but are not used by the executed training or feature-extraction functions. `--mode` controls the model-checkpoint filename prefix.

**Important:** Before running `run.py`, update the absolute output path on the `np.savetxt(...)` line near the end of that file. As supplied, it points to a dataset-specific directory on the original workstation.

## Outputs

| Output | Format | Description |
| --- | --- | --- |
| `<mode>_ResNet18_params.pth` | PyTorch checkpoint | Trained ResNet-18 parameters. |
| `epoch_avg_loss.txt` | Tab-separated text | Mean training loss per epoch. |
| Location specified by `np.savetxt(...)` in `run.py` | Text matrix | Min–max-scaled, 20-dimensional UMAP embeddings; one row per ion image. |

The intermediate **512-dimensional features** are calculated in `DeepION_predicting()` and passed to `DimensionalityReduction()`, but they are not separately saved by the current command-line script.

## Settings used by the supplied scripts

| Setting | Value |
| --- | --- |
| Encoder | ImageNet-pretrained ResNet-18 |
| Training epochs | 50 |
| Nominal batch size | 100 (last training batch may be smaller) |
| Loss balance `lambda` | 0.3 |
| Optimizer | Adam, learning rate `2e-4` (other parameters use PyTorch defaults) |
| Random seed | 50 in `DeepION.py` |
| Feature dimension | 512 |
| UMAP dimension / metric / random state | 20 / cosine / 0 |
| Target encoder | Momentum/EMA target encoder enabled in `byol_pytorch.py` |

## Implementation notes and current limitations

- **GPU selection:** `DeepION.py` and `neg_augment.py` currently select CUDA device index `2`, so the code assumes that GPU is accessible. Adapt this setting for a different workstation.
- **Partial batches and NAV:** In the supplied `neg_augment.py`, intensity inversion is executed only when the batch size is exactly `100` and the image height matches `target_height`. Smaller final training batches are returned without inversion. This is a known limitation of the current scripts.
- **Feature-extraction batch handling:** The supplied `DeepION_predicting()` code assumes a nonempty remainder after full 100-ion batches; datasets with fewer than 100 ions or an ion count divisible by 100 can encounter an error.
- **Local paths:** `DownStreamTask.py` and the embedding-output line in `run.py` contain dataset-specific absolute paths that must be adapted.
- **Downstream analyses:** These six files do not include complete, standalone scripts for positive/negative ion retrieval, ion clustering, or subregion-specific molecular-network construction.
- **Validation:** These files have been documented based on source inspection; no end-to-end execution or reproduction of the paper's figures is claimed here.

## Citation

If you use DeepCoEx, please cite:

Xu, J.; et al. **Beyond Co-Localization: DeepCoEx Learns Exclusion-Aware Spatial Representations in Mass Spectrometry Imaging via Dual-Contrastive Learning.** *Analytical Chemistry*. DOI: [10.1021/acs.analchem.6c02338](https://doi.org/10.1021/acs.analchem.6c02338).
