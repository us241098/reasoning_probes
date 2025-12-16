# Reasoning Error Detection via Hidden State Probing: A Feature Ablation Study


---

## Introduction

Large language models often generate step-by-step reasoning traces when solving mathematical problems, but these traces may contain errors at intermediate steps. Detecting such errors is crucial for improving model reliability and interpretability. We investigate whether linear probes trained on hidden states from transformer layers can predict step-level correctness in reasoning traces.

We conduct a feature ablation study comparing five different feature configurations: (1) multi-layer hidden states (last token only), (2) multi-layer hidden states (mean pooled), (3) attention pattern features, (4) token logits/probability features, and (5) a combination of all features. We train probes on GSM8K and evaluate cross-dataset generalization on AQuA.

## Method

### Reasoning Generation

We use `Qwen2.5-0.5B-Instruct` to generate step-by-step reasoning traces for mathematical problems. The model is prompted to solve problems explicitly, showing calculations at each step and providing a final answer in a structured format. We use greedy decoding for deterministic outputs and process samples in batches for efficiency.

### Process Reward Model Labeling

We employ ThinkPRM-1.5B as a Process Reward Model (PRM) to automatically label each reasoning step as correct or incorrect. The PRM analyzes the reasoning trace and provides step-level scores. We use a strict prompting strategy to ensure the PRM penalizes invalid reasoning steps early.

### Feature Extraction

We extract five types of features from the model's internal representations:

**Multi-layer Hidden States:** We extract hidden states from multiple transformer layers (1/4, 1/2, 3/4, and final layer) corresponding only to the tokens of the current reasoning step. For experiment 1, we use only the last token's representation from each layer. For experiment 2, we mean-pool across all tokens in each reasoning step.

**Attention Patterns:** We aggregate attention weights from the final layer across attention heads, computing: (1) mean attention from step tokens to all tokens, (2) standard deviation of attention, (3) mean attention to question tokens, and (4) mean attention to answer tokens.

**Token Logits/Probabilities:** We extract confidence measures from the model's output logits: (1) mean maximum probability across step tokens, (2) standard deviation, and (3) minimum confidence.

### Linear Probe Training

We train logistic regression classifiers (linear probes) on the extracted features to predict step correctness. The probe is trained on GSM8K reasoning traces and evaluated on GSM8K and AQuA to assess cross-dataset generalization.

## Experimental Setup

### Datasets

- **Training:** GSM8K (1024 samples)
- **Evaluation:** GSM8K (200 samples), AQuA (200 samples)

### Feature Configurations

We run five experiments:

1. Multi-layer hidden states (last token only): 4480 dimensions
2. Multi-layer hidden states (mean pooled): 3584 dimensions  
3. Attention patterns only: 4 dimensions
4. Token logits/probabilities only: 3 dimensions
5. All features combined: 8071 dimensions

### Evaluation Metrics

We report Area Under the ROC Curve (AUC) and accuracy for each experiment on the evaluation datasets.

## Results

### Training and Validation Performance

Table 1 shows the performance for each feature configuration.

**Table 1: Experiment Results (AUC / Accuracy)**

| Experiment | Feature Dim | GSM8K (Val Split) | AQuA (OOD) |
|------------|-------------|-------------------|------------|
| Exp 1: Hidden States (Last Token) | 4480 | 0.683 / 0.639 | 0.598 / 0.643 |
| Exp 2: Hidden States (Mean Pooled) | 3584 | **0.745** / **0.687** | **0.634** / 0.583 |
| Exp 3: Attention Only | 4 | 0.500 / 0.543 | 0.500 / 0.302 |
| Exp 4: Logits Only | 3 | 0.534 / 0.549 | 0.466 / 0.447 |
| Exp 5: All Features | 8071 | 0.719 / 0.650 | 0.612 / **0.651** |

### Key Findings

- **Hidden States are Essential:** Experiments 1, 2, and 5 (which include hidden states) significantly outperformed baselines.
- **Mean Pooling is Superior:** Mean Pooled features (Exp 2) achieved the highest AUC on both GSM8K (0.745) and AQuA (0.634), suggesting that aggregating information across the entire reasoning step captures the error signal better than just the final token.
- **All Features Combined:** Adding attention and logits (Exp 5) did not improve performance over Mean Pooling alone (0.719 vs 0.745 AUC on GSM8K), likely due to the noise from the ineffective features or dimensionality issues.
- **Attention and Logits:** These features alone performed poorly, often near random guessing (0.5 AUC).

## Conclusion

Our feature ablation study reveals that **mean-pooled hidden states** provide the most robust signal for *ranking* error likelihood (highest OOD AUC), while combining all features yields the best *classification accuracy*. Simple scalars like logits and attention statistics were insufficient for this task.

### Robustness Improvements
During this study, we implemented several robustness measures that improved data quality:
1.  **Strict PRM Prompting**: We enforced a strict "examiner" persona on the PRM to catch structural errors early, reducing false positives in step labeling.
2.  **Robust Parsing**: We implemented logic to handle complex answer formats (e.g., AQuA's multiple choice options, LaTeX fractions in MATH problems) to ensure accurate ground-truth comparison.
3.  **Data-Parallel Processing**: We scaled batch processing to handle larger datasets efficiently without concurrency issues.

### Limitations
While effective, our approach has limitations:
1.  **Step Alignment Noise**: We split reasoning traces by regex (`Step N:`), which may not perfectly align with the model's internal "thought boundaries." This can introduce noise into the hidden state extraction.
2.  **PRM Hallucination**: Despite strict prompting, the PRM may occasionally "hallucinate" verification for long traces, marking steps as correct/incorrect simply because they look plausible.
3.  **Linear Probe Simplicity**: We use a simple logistic regression. While this shows that error information is *linearly* accessible, non-linear probes (MLPs) might extract more subtle signals.

### Future Work
- **Head-Specific Probing**: Instead of aggregating attention statistics, training probes on specific attention heads (e.g., "induction heads" or "error-detection heads") could yield more precise detectors.
- **Better Hidden Representations**: Extracting features from different token positions (e.g., periods, newlines) or using more sophisticated methods could capture better signals than simple mean pooling.
- **Non-Linear Probes**: Investigating whether shallow MLPs can significantly outperform linear probes would determine if the error signal is complex or simple.
- **Larger Models**: Testing on larger reasoning models (e.g., Qwen-7B or Llama-3-8B) to see if error representations become more distinct with scale.
- **Contrastive Training**: Generating pairs of correct/incorrect steps for the *same* problem state could provide a cleaner training signal for the probe.



## Setup

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster processing, but CPU will work)
- At least 8GB RAM (16GB+ recommended)
- ~5GB disk space for model downloads and results

### Installation

1. **Clone or navigate to the repository directory**

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```
### Basic Usage

Run all five feature ablation experiments:

```bash
python main_experiments.py
```

This will:
1. Generate reasoning traces for training data (GSM8K, ~1024 samples)
2. Train linear probes for each of the 5 feature configurations
3. Generate evaluation traces (GSM8K test + AQuA test)
4. Evaluate all probes on evaluation datasets
5. Save results to `results_experiments/`

**Expected runtime:** ~3-4 hours on a GPU (depending on hardware), significantly longer on CPU.

### Command-Line Options

**Debug mode** - Run only the first experiment (faster for testing):
```bash
python main_experiments.py --debug-single-exp
```

**Use existing traces** - Skip trace generation and use previously saved traces:
```bash
python main_experiments.py --use-existing-traces
```

This is useful if you want to:
- Re-run experiments with different feature configurations without regenerating traces
- Test different probe architectures on the same data
- Save time during development iterations

### Output Files

The script generates the following files in `results_experiments/`:

1. **`all_reasoning_traces.json`** - Complete reasoning traces for all datasets (training + evaluation)
   - Contains question, expected answer, model answer, step-by-step reasoning, PRM scores, and labels
   - Format matches the trace structure used in the main pipeline

2. **`exp{N}_results.json`** - Individual experiment results (5 files total)
   - Experiment configuration and feature dimensions
   - Training statistics (sample counts, label distribution)
   - Probe performance (train/validation AUC and accuracy)
   - Evaluation metrics on GSM8K and AQuA

3. **`exp{N}_traces.json`** - Reasoning traces for each experiment (same format as `all_reasoning_traces.json`)

4. **`all_experiments_summary.json`** - Aggregated summary of all experiments
   - Timestamp and configuration
   - Complete results table for easy comparison