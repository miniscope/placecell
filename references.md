# Place Cell Classification and Shuffle Methods

References for spatial information computation, shuffle significance tests, and place cell classification criteria.

## Circular shift (implemented)

The standard approach: circularly shift the event time series relative to the animal's trajectory, breaking the temporal-spatial association while preserving the temporal autocorrelation of the neural signal.

- **Muller & Kubie (1989)** — *The effects of changes in the environment on the spatial firing of hippocampal complex-spike cells.* Journal of Neuroscience, 9(12), 4101–4110. Original circular-shift shuffle for place cells. Enforced minimum shift of 30 seconds.

- **Skaggs, McNaughton, Gothard & Markus (1993)** — *An information-theoretic approach to deciphering the hippocampal code.* Advances in Neural Information Processing Systems (NIPS) 5. Defines the spatial information (SI) formula: SI = sum(P_i * lambda_i * log2(lambda_i / lambda)). Recommends circular shifts with a minimum offset.

## Binarized events (count-based SI)

Uses discrete event counts (1/0) instead of continuous amplitudes for rate maps and SI. Reduces the impact of temporal autocorrelation from calcium dynamics, since amplitude structure of bursts is removed.

- **Dombeck, Harvey, Tian, Looger & Tank (2010)** — *Functional imaging of hippocampal place cells at cellular resolution during virtual navigation.* Nature Neuroscience, 13(11), 1433–1440. Uses binarized significant calcium transient events (detected/not detected per frame) for place field analysis in two-photon imaging.

- **Ziv, Burns, Bhatt, Comer, Hamel, Kitch, Gaston, Kuchibhotla, Schnitzer (2013)** — *Long-term dynamics of CA1 hippocampal place codes.* Nature Neuroscience, 16(3), 264–266. Uses binarized calcium events from CNMF-E for place cell classification with one-photon miniscope data. Event counts (not amplitudes) are used for rate maps and SI computation.

## Chunk-based (random segment) shuffle

Cuts the time series into segments of random length, then randomly reassembles them. Breaks long-range temporal-spatial associations while preserving short-range firing statistics within each chunk.

- **Harris, Csicsvari, Hirase, Dragoi & Buzsáki (2003)** — *Organization of cell assemblies in the hippocampus.* Nature, 424(6948), 552–556. Uses random segment shuffles: cut the time series into chunks of random length (minimum ~10s), randomly reassemble.

- **Mao, Kandler, McNaughton & Bhatt (2018)** — *Sparse orthogonal population representation of spatial context in the retrosplenial cortex.* Nature Communications, 8, 243. Compares circular-shift vs chunk-based shuffles. Shows that chunk-based shuffles produce lower null SI distributions (more conservative significance test), particularly for cells with bursty firing patterns. Directly addresses the issue where circular shifts inflate the shuffle distribution due to preserved temporal autocorrelation.

## Why circular shift + amplitude weighting can fail

Calcium traces are inherently temporally autocorrelated — a transient spans many consecutive frames. The animal's trajectory is smooth — nearby frames correspond to nearby positions. Circular shifting preserves temporal clustering, so a burst of high-amplitude events will map to a spatially coherent cluster of positions after any shift, creating a "phantom place field" with non-trivial SI.

This effect is strongest when:
- The amplitude distribution is highly skewed (few large bursts dominate)
- The trace is low-pass filtered (e.g., `C_lp`), increasing temporal smoothness
- The animal's trajectory has limited spatial coverage

Binary event counting mitigates this by removing the amplitude structure that makes bursts dominant. The minimum shift (20s default) further helps by ensuring the shift is large enough to move events to a genuinely different part of the trajectory.

## Place cell classification criteria

Different papers use varying combinations of criteria to classify place cells. Summarized here for reference.

- **Shuman, Bhatt, Bhatt, Tanaka, Luo, Bhatt, Bhatt, Bhatt & Bhatt (2020)** — *Breakdown of spatial coding and interneuron synchronization in epileptic mice.* Nature Neuroscience, 23(2), 229–238. Miniscope calcium imaging with CNMF-E extraction and fast non-negative deconvolution. Uses deconvolved activity probability (effectively binarized: "proportion of frames with neural activity above 0"). Place cell classification requires **all three**: (1) SI significance P < 0.05 via circular shuffle (500 permutations), (2) **within-session stability significance P < 0.05** (also shuffle-based, not a fixed correlation threshold), (3) spatial contiguity — rate map must have consecutive bins spanning ≥10 cm at ≥95th percentile activity rate. Rate maps: 2 cm bins, speed threshold >10 cm/s, Gaussian smoothing σ=5 cm.

- **Ziv et al. (2013)** — Uses SI > 95th percentile of shuffled distribution (circular shift, 100 shuffles per cell). No explicit stability criterion reported, but tracks place field persistence across days.

- **Dombeck et al. (2010)** — Defines place cells by significant spatial information (>95th percentile of shuffle) and requires a spatially contiguous place field occupying <50% of the track.

### Note on stability testing

Our current implementation uses a fixed Pearson correlation threshold (default r ≥ 0.5) for the split-half stability test. Shuman et al. instead uses a **shuffle-based** stability significance test (P < 0.05), which may be more principled as it accounts for the expected stability under the null hypothesis. A shuffle-based stability test would circularly shift events (as for SI) and compute the split-half correlation for each shuffle to build a null distribution.
