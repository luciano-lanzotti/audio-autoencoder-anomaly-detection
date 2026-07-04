# Audio Autoencoder for Class Separation & Anomaly Detection

A 1D convolutional autoencoder that learns to distinguish three audio types — Korean children's songs, Arabic speech, and bird sounds — purely from raw waveforms, and uses the resulting latent space to classify unseen test tracks and flag anomalies.

## Course & Authorship

- **Author**: Luciano Lanzotti
- **Course**: AI Methods and ML for Physics
- **Program**: Bachelor's Degree in Physics — Sapienza University of Rome
- **Professor**: [Professor Name]
- **Academic year**: [Academic Year]

## Project Overview

The task was to recognize three classes of 4.0-second audio recordings, sampled at 16 kHz:

1. Korean children's songs
2. Arabic speech
3. Bird sounds

using a labeled training set (`gruppo_0`, ~450 tracks, ~150 per class) and to then apply the trained model to a separate, unlabeled test set (`gruppo_4`, 71 tracks) in order to (a) predict each test track's class and (b) detect anomalies — tracks that don't belong to any of the three known classes.

Rather than a supervised classifier, the approach is unsupervised: an autoencoder is trained purely to reconstruct audio fragments, and the compressed latent representation it learns along the way turns out to separate the three classes well enough to support both classification (via k-NN on the latent space) and anomaly detection (via clustering).

## Repository Contents

| File | Description |
|---|---|
| `audio_autoencoder_anomaly_detection.ipynb` | Main notebook: preprocessing, model definition, training, latent-space analysis, and anomaly detection. |
| `salvataggi/22_giugno/1/model_ae.pt` | Trained autoencoder checkpoint (whole model object, loadable via `torch.load`), matching the path the notebook expects. |
| `salvataggi/22_giugno/1/loss_results_model_ae.txt` | Per-epoch training/validation loss history (400 epochs). |
| `my_labels_0.csv` | Filename → class label (0 = bird, 1 = Korean, 2 = Arabic) mapping for the 450 training tracks. |
| `label_predette.csv` | Final predicted class (or `-1` for anomaly) for each of the 71 test tracks. |
| `requirements.txt` | Python dependencies. |

## Data Availability

The raw `.wav` audio datasets referenced by the notebook are **not included** in this repository (due to size):

- `gruppo_0/` — ~445 valid training tracks (4.0 s, 16 kHz, mono), matched to `my_labels_0.csv`.
- `gruppo_4/` — 71 unlabeled test tracks (4.0 s, 16 kHz, mono), used for anomaly detection and class prediction.

To reproduce the preprocessing and training cells from scratch, place these folders alongside the notebook with the same names. Without them, the preprocessing/training cells will not run — but the **trained model checkpoint is included** and resolves correctly at the path the notebook expects (`salvataggi/22_giugno/1/`), so the model-loading and downstream analysis cells (latent-space visualization, anomaly detection, classification) can still be run against a freshly loaded model, provided the fragmented input tensors are available.

## Method

**Preprocessing**: each track is loaded, filtered to keep only the expected shape (1, 64000 samples), min-max normalized to [0, 1], and split into 8 non-overlapping fragments of 8000 samples (0.5 s) each. Training fragments are split 70/30 into train/validation sets (batch size 64).

**Model — 1D convolutional autoencoder** (38,513 trainable parameters):

| Layer | Kernel Size | Stride | Dilation | Padding | Output Padding |
|---|---|---|---|---|---|
| Conv1d-1 | 40 | 2 | 1 | 1 | – |
| Conv1d-2 | 30 | 3 | 1 | 1 | – |
| Conv1d-3 | 20 | 4 | 2 | 1 | – |
| Conv1d-4 | 10 | 6 | 2 | 1 | – |
| Conv1d-5 | 5 | 7 | 2 | 1 | – |
| ConvTranspose1d-1 | 5 | 7 | 2 | 0 | 2 |
| ConvTranspose1d-2 | 10 | 6 | 2 | 0 | 0 |
| ConvTranspose1d-3 | 20 | 4 | 2 | 0 | 0 |
| ConvTranspose1d-4 | 30 | 3 | 1 | 1 | 0 |
| ConvTranspose1d-5 | 40 | 2 | 1 | 1 | 0 |

Each encoder layer is followed by batch normalization and a ReLU activation; each decoder layer uses ReLU except the last, which uses a sigmoid (matching the [0, 1]-normalized input). Channel counts grow 1→4→8→16→32→64 through the encoder and mirror back down through the decoder.

Design rationale:
- **Kernel size** shrinks (40 → 5) across the encoder: a large receptive field early on captures global track structure, while smaller kernels later suit the already-reduced sequence length.
- **Stride** starts small (2) for a controlled initial reduction, then increases in later layers to progressively select coarser, more important features.
- **Dilation** is 1 in the first two layers (to avoid losing fine-grained signal detail) and increases to 2 afterward (to grow the receptive field where the sequence is already short).

Unusually, the bottleneck is **not** flattened into a vector: the latent representation of each 8000-sample fragment is a `(64, 7)` matrix (64 channels × 7 time steps). Keeping this 2D structure — rather than collapsing it — was found to improve reconstruction quality. For visualization and clustering, this matrix is pooled (via `adaptive_avg_pool1d`) into a 64-dimensional vector per fragment.

**Training**: Huber loss, Adam optimizer (lr = 1e-3, with weight decay), 400 epochs, 70/30 train/validation split.

**Latent-space analysis**: after training, UMAP reduces the pooled 64-dimensional latent vectors to 2D for visualization — both at the fragment level and at the whole-track level — showing three well-separated clusters corresponding to the three classes.

**Anomaly detection** on the 71-track test set was attempted with two methods:
1. **Whole-track MSE thresholding** (flagging tracks with reconstruction error beyond mean + 3σ) — this was tried and **rejected**, since the reconstruction error on the test set (including on true anomalies) turned out to be comparable to the training/validation error, making it a poor discriminator.
2. **t-SNE + KMeans clustering of the latent space** — adopted instead. Reducing the test set's latent vectors with t-SNE reveals three clusters, one noticeably smaller and more separated than the other two; fragments in that small cluster are treated as anomalous, and any track with at least one anomalous fragment is flagged.

**Classification**: a k-NN classifier is trained on the labeled validation set's latent vectors (best k = 3, chosen via 5-fold cross-validation) and used to predict each test track's class from its pooled latent vectors; tracks already flagged as anomalous by the clustering method are overridden with label `-1`.

## Results

- k-NN classification accuracy (5-fold cross-validation, k = 3): **A = 0.859 ± 0.013**
- Anomaly detection (latent-clustering method) flagged 8 of the 71 test tracks. Manual listening confirmed these as: 6 rock songs, 1 silent/mute track, and 1 Korean song (a false positive).
- Final per-track predictions (class 0/1/2, or -1 for anomaly) are saved in `label_predette.csv`.

## How to Run

This notebook was developed and run on **Google Colab** with GPU acceleration, and has not been re-run or verified locally as part of preparing this repository. To use it:

1. Open `audio_autoencoder_anomaly_detection.ipynb` in Google Colab (recommended) or a local Jupyter environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Provide the `gruppo_0/` and `gruppo_4/` audio folders (see [Data Availability](#data-availability)) if you want to run preprocessing/training from scratch; otherwise, the included checkpoint can be loaded directly for downstream analysis.

## Known Limitations

- The raw audio datasets are not distributed with this repository.
- The model is saved as a whole pickled object (`torch.save(model, ...)`) rather than a `state_dict`, which is less portable across PyTorch/library versions.
- The original training run was not seeded, so exact loss curves are not perfectly reproducible from scratch (seeds are only fixed for the downstream analysis steps, after loading the saved checkpoint).
- The "anomalous cluster" in the latent-space clustering method is identified by visual inspection (smallest/most separated cluster) rather than a fully automated criterion.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
