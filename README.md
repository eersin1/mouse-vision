# Decoding natural images from mouse visual cortex

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eersin1/mouse-vision/blob/main/mouse_vision.ipynb)

A mouse looks at a photograph while electrodes record its visual cortex. I decode that activity
into five different descriptions of the same image: raw pixels, three depths of a CNN, and CLIP.
Every test image is one the decoder has never seen.

The result isn't that decoding works. It's *which* descriptions work.

![What kind of visual information can be read out of mouse visual cortex?](figures/feature_spaces.png)

## Result

2,968 visual-cortex neurons pooled from 8 mice, 118 natural scenes, 250 ms of spiking per trial.
Six-fold cross-validation over images, so every scored image was absent from training.

| description of the image | among the 20 held out | among all 118 |
|---|---|---|
| raw pixels | 0.822 | **0.341** |
| early vision (edges, texture) | **0.874** | 0.279 |
| mid-level | 0.795 | 0.085 |
| high-level object parts | 0.736 | 0.000 |
| semantic (CLIP) | 0.271 | 0.034 |

Chance is 0.050 and 0.008.

Accuracy falls steadily as the target gets more semantic. Two parts of that are worth pulling out.

**Early-vision features beat raw pixels.** That's a claim about what the cortex represents, not
just that the images are recoverable.

**CLIP sits near chance on the 118-way test.** Reconstructing images through CLIP is the standard
recipe in human fMRI work ([MindEye](https://arxiv.org/abs/2305.18274) and successors), and it
assumes the recorded cortex carries semantic structure. On this data it doesn't.

![Held-out images, decoded and matched](figures/retrieval.png)

Top row is the photograph shown. Middle is what the neurons alone predict: blurry, but it carries
the coarse layout. Bottom is the image the decoder picked out of 20 candidates, green where correct.

![How many neurons do you need?](figures/neuron_scaling.png)

Accuracy is still climbing at 2,968 neurons, so the limit here is how many neurons were recorded,
not the decoder.

## Method

- **Data:** [Allen Brain Observatory, Visual Coding Neuropixels](https://portal.brain-map.org/explore/circuits/visual-coding-neuropixels)
  (CC BY 4.0). 8 sessions, filtered to well-isolated units in `VIS*` areas using the Allen
  Institute's recommended quality thresholds.
- **Spike counts** in a 30 to 250 ms window after stimulus onset, square-root transformed to
  stabilise Poisson variance.
- **Pseudo-population:** every mouse saw the same 118 images, so trials of the same image are
  paired across animals into one population.
- **Ridge regression** from neural activity to image description, `alpha` swept over
  `[1e3, 1e4, 3e4, 1e5]`.
- **Scored by retrieval:** cosine similarity between predicted and true descriptions, counted as a
  hit when the nearest one is correct.

The split is over images, not trials. Splitting trials would leave the same photograph on both
sides, which measures memorisation instead of generalisation.

## Limitations

- Pseudo-populations pool neurons from different animals recorded at different times. Nothing here
  speaks to correlations *between* neurons.
- 118 images is a small stimulus set, and the generalisation claim is bounded by it.
- Decoded images carry coarse layout, not detail (pixel correlation around 0.44).
- The stimuli are grayscale and low-resolution while CLIP was trained on colour photographs. Some
  of CLIP's poor showing may be the stimuli rather than the mouse. That's the obvious next test.
- `alpha` was tuned on the same cross-validation used to report results, which slightly favours the
  tuned numbers.

## Prior work

Mouse visual decoding is an established field and nothing here is methodologically novel. The data
is the Allen Institute's. The CLIP-reconstruction approach being tested comes from human fMRI work,
chiefly MindEye and its successors. What's mine is the comparison across description types on this
dataset, and the negative result for semantic features.

## Running it

Open in Colab with the badge above, or locally:

```bash
pip install numpy pandas scipy scikit-learn matplotlib h5py torch torchvision open_clip_torch pillow jupyter
jupyter notebook mouse_vision.ipynb
```

The notebook downloads what it needs. Session files are ~2.9 GB each and each one is deleted right
after it's processed, so peak disk stays near 3 GB instead of 23 GB. The derived caches total under
100 MB.

First run takes roughly 45 to 80 minutes, almost all of it download time. After that everything
runs off the cache in seconds.
