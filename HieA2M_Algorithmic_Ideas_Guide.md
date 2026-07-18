# HieA2M Algorithmic Ideas Guide

This guide maps the algorithmic innovations of HieA2M to their specific code implementations. The goal is to provide a theoretical understanding of *why* the codebase was modified in the way detailed in the Code Implementation Guide, rooted in resolving the seven theoretical misalignments between graph-based spatial representations (HieA2G) and fuzzy temporal representations (Generalized Moment Retrieval).

---

## 1. The Core Bottleneck: From BCE Existence to AMC

**The Baseline Problem**: The official FlashVTG-GMR baseline relies on a naive binary existence head (`exist_head` using `sigmoid` and BCE Loss). Our score drift analysis revealed that this head suffers from extreme score inflation (all scores drift above the hard 0.4 threshold) because BCE lacks class balancing, the validation set ignored negatives, and the representation space couldn't adequately separate null-sets from positives.

### Innovation: Adaptive Moment Counter (AMC)
HieA2M completely removes the binary gate and replaces it with a 5-class Adaptive Moment Counter (`{0, 1, 2, 3, 3+}`). This tackles multiple misalignments:

#### Misalignment #7: Distribution-Aware Features
*   **Theory**: In spatial tasks, you count discrete objects. In temporal tasks, proposals overlap heavily. Simply passing global video/text features to a counter ignores the temporal density.
*   **Implementation (`model.py` -> `AdaptiveMomentCounter`)**: The AMC explicitly takes a 10-bin histogram of `window_scores` and projects it to a $D$-dimensional feature. It also calculates a $N \times N$ temporal distance matrix ($1 - \text{tIoU}$) among proposals, interpolates it, and adds it as a feature. This gives the counter explicit geometric and density cues about the temporal landscape.

#### Misalignment #4: Penalizing Jump Errors
*   **Theory**: Standard cross-entropy penalizes all wrong classes equally. Predicting "0 moments" when the truth is "2 moments" is a much worse error than predicting "1 moment".
*   **Implementation (`model.py` -> `loss_exist`)**: We implemented an **Ordinal Regression Loss**. It calculates the expected count value $\sum (P(c) \times c)$ and computes the MSE against the true count label. Crucially, this is applied *only* to positive samples to avoid biasing the model toward 0 (since 51% of the dataset is null).

#### Null-Set Leakage Prevention
*   **Theory**: Because the model is heavily exposed to positive examples, the feature representations for null-set queries tend to "leak" toward single or multi-moment classes.
*   **Implementation (`model.py` -> `loss_exist`)**: We introduced the **Null Anchor Loss**. For null-set samples, it enforces a hard margin: $P(\text{class} \ge 2) - P(\text{class} == 0) + 0.3 \le 0$. This explicitly pushes the representation of empty events away from multi-moment events.

---

## 2. Hierarchical Text-Moment Alignment (HTMA)

The AMC determines *how many* moments exist. The HTMA provides dense auxiliary supervision to ensure the text and video representations are strictly aligned across three granularities.

### Misalignment #1 & #3: Word-Frame Recovery & Margin Loss (HTMA Level 1)
*   **Theory**: If a query asks for a "goal", the model should be able to reconstruct the word "goal" solely from the relevant video frames. HieA2G completely skipped null-set samples for this task. However, in GMR, telling the model *not* to find a word in a null-video is highly informative.
*   **Implementation (`dataset.py` & `model.py`)**: 
    1.  We implemented online probabilistic masking in `dataset.py`, deliberately targeting action words (e.g., "saves", "passes").
    2.  In `loss_mask_rec`, for positive samples, we use standard Cross Entropy to recover the word. For **null-set samples**, we use a **Margin Loss** ($m=0.5$). The model is explicitly trained to ensure the logit for the *correct* word is lower than other words, enforcing that the action is genuinely missing from the video.

### Misalignment #2: Phrase-Segment Alignment with Optimal Transport (HTMA Level 2)
*   **Theory**: A query usually contains multiple phrases (e.g., "The player passes" and "the goalkeeper saves"). Bipartite matching (used in DETR/HieA2G) forces a rigid 1-to-1 mapping. Temporal events are fuzzy; a phrase might map to multiple short moment proposals, or vice-versa.
*   **Implementation (`model.py`)**: We replaced bipartite matching with **Sinkhorn Optimal Transport**. `SinkhornMatcher` computes a soft transport plan $T$ between phrase embeddings and moment embeddings, allowing many-to-many alignment. The loss is computed as $-\sum (T \cdot \log(\text{sim}))$.

### Global Alignment (HTMA Level 3)
*   **Theory**: The highest level ensures the overall sentence semantics match the overall video context.
*   **Implementation (`model.py` -> `loss_global_align`)**: A standard symmetric InfoNCE loss between mean-pooled text and video features, projected to 128 dimensions.

---

## 3. Inference Logic: Adaptive Thresholding

### Misalignment #6: The Failure of Hard Thresholds
*   **Theory**: The baseline FlashVTG-GMR relies on a hard threshold ($\tau=0.4$) to gate predictions. However, different queries exhibit vastly different score distributions. Some peak at 0.9, while fuzzy/difficult queries might peak at 0.5. A hard threshold of 0.7 would arbitrarily eliminate correct predictions for difficult queries.
*   **Implementation (`inference.py` & gating CLI)**: The AMC dictates the strategy:
    *   If AMC predicts `0`, return the empty set immediately.
    *   If AMC predicts `1, 2, 3`, select the Top-K moments after NMS.
    *   If AMC predicts `3+` (multi-moment), use **Relative Peak Filtering**. The threshold is computed dynamically relative to the highest scoring moment: $\tau^* = \max(0.5, 0.8 \times \text{top-1 score})$. This ensures robust retrieval regardless of score diffusion.

---

## 4. Addressing the Score Drift Problem

Our internal diagnostics revealed that the baseline `pred_exist_score` values consistently drifted upwards. The HieA2M methodology addresses this at the root:

1.  **Class Imbalance**: The dataset is 51.3% null, 30.4% single, 18.3% multi. The baseline BCE loss ignored this. HieA2M applies a **Focal Loss** ($\alpha=[0.5, 0.7, 0.9, 0.95, 0.97]$) inside `loss_exist` to force the network to pay attention to the harder multi-moment classes.
2.  **Soft Labels for Ambiguity**: Temporal boundaries are inherently ambiguous. A single 10-second pass might be annotated as two 5-second passes by a different annotator. We use `count_soft` labels (e.g., `[0.0, 0.2, 0.6, 0.2, 0.0]`) via a KL-divergence-like loss to prevent the network from being overconfident on noisy annotations, stabilizing the gradients.
3.  **Loss Weight Scaling**: In the baseline, the existence loss weight was 1.0, while span/label losses were 10.0/4.0, drowning out the null-set prediction signal. The HieA2M composite `loss_exist` effectively produces a gradient magnitude comparable to the main task losses, ensuring the AMC trains effectively.
