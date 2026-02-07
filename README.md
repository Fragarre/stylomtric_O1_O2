# Tacitus Authorship Verification  
**Stylometric comparison of *Dialogus de Oratoribus* with Tacitean works**

## 1. Objective

This project investigates the authorship of *Dialogus de Oratoribus* through **stylometric authorship verification**, comparing it against a reference corpus of undisputed works by Tacitus (*Germania, Agricola, Historiae, Annales*, etc.).

Rather than classifying texts among many candidates, the focus is on **verification**:

> *Is the style of the Dialogus compatible with Tacitus’ undisputed works?*

To answer this, the project implements two well-established verification approaches:
- **O1 (Profile-based verification with internal validation)**
- **O2 (General Imposters method)**

The analysis is deliberately conservative and methodologically transparent, following practices common in computational stylometry and digital philology.

---

## 2. Methodological Overview

### 2.1 Unified Feature Representation

The stylistic signal is captured by **combining two complementary feature types**:

- **Word unigrams**  
  Capture lexical preferences and function-word usage.

- **Character trigrams**  
  Capture morphological, orthographic, and rhythmic patterns that are more robust across genres and topics.

Both representations are:
- Frequency-based
- L1-normalized
- Concatenated into a **single unified feature space**

This combination balances interpretability (word features) and robustness (character features), a common strategy in authorship studies of Latin and historical texts.

---

### 2.2 Optional Text Segmentation

The script supports **optional segmentation** of long texts into fixed-size chunks (default: 1500 words).

- `USE_SEGMENTATION = True`  
  Treats each segment as an independent stylistic sample

- `USE_SEGMENTATION = False`  
  Treats each work as a single unit

Segmentation improves statistical stability and allows internal validation even with a limited number of works.

---

## 3. Verification Models

### 3.1 O1 – Profile-Based Verification with Internal Validation

**Step 1: Internal validation (leave-one-out)**  
Each Tacitean text (or segment) is:
- Removed from the corpus
- Compared against a profile built from the remaining Tacitus texts

This produces a **distribution of internal stylistic distances**, representing normal variation *within* Tacitus.

**Step 2: Unknown text comparison**  
*Dialogus de Oratoribus* is compared to the same Tacitus profile.

**Interpretation principle**:
- If the Dialogus distance falls **inside or close to** the Tacitean internal distribution → stylistically compatible
- If it falls clearly outside → stylistic divergence

Two visualizations are produced:
- Boxplot (internal Tacitus vs Dialogus)
- Histogram with Dialogus marked

---

### 3.2 O2 – General Imposters Method

O2 addresses a key weakness of profile-based methods:  
*How distinctive is the author relative to others?*

**Procedure**:
1. Randomly sample:
   - A subset of Tacitus texts
   - A subset of impostor texts (other Latin authors)
2. Build two profiles:
   - Tacitus profile
   - Impostor profile
3. Compare which profile is closer to the Dialogus
4. Repeat many times (default: 1000 iterations)

**Output**:
- A probability score between 0 and 1

**Interpretation**:
- Values close to **1.0** → Dialogus consistently closer to Tacitus
- Values around **0.5** → inconclusive
- Values close to **0.0** → closer to impostors

A bar chart summarizing the O2 score is saved.

---

## 4. Code Structure

Main components of the script:

- **Data loading**
  - Reads `.txt` files from corpus folders
- **Preprocessing**
  - Optional segmentation
  - Unified vectorization (word + char n-grams)
- **O1 verification**
  - Internal validation
  - Unknown text comparison
- **O2 verification**
  - Iterative imposters framework
- **Visualization**
  - Automatically saves high-resolution JPG figures

All vectorizers are fitted **only on training data**, preventing information leakage.

---

## 5. Output and Results

The script produces:

- `o1_tacitus_dialogus_boxplot.jpg`  
  Internal Tacitus distances with Dialogus highlighted

- `o1_tacitus_dialogus_histogram.jpg`  
  Distribution of Tacitean stylistic variation

- `o2_tacitus_dialogus.jpg`  
  O2 probability score

Console output includes:
- Dialogus distance to Tacitus profile
- Mean and standard deviation of Tacitus internal distances
- O2 probability score

These results are intended for **interpretive comparison**, not binary classification.

---

## 6. How to Use

1. Place the corpora in the expected folder structure:
   - Tacitus (undisputed works)
   - Impostor authors
   - Dialogus text
2. Adjust the segmentation parameter if desired
3. Run the script
4. Inspect:
   - Visualizations
   - Numerical outputs
5. Interpret results comparatively, not in isolation

---

## 7. Scope and Extensions

This version treats **all Tacitus texts as a single authorial corpus**.

Possible extensions include:
- Comparison of *Dialogus* with each Tacitean work individually
- Genre-controlled imposters
- Feature ablation studies
- Bootstrap confidence intervals

These extensions are intentionally left modular and can be built directly on the current codebase.

---

## 8. Scholarly Note

This project is not intended to “prove” authorship, but to provide **quantitative stylistic evidence** that can be integrated with philological, historical, and rhetorical analysis.

Stylometry complements — it does not replace — traditional scholarship.
