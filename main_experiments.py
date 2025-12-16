"""
Run feature ablation experiments to see which features work best for detecting reasoning errors.

We test 5 different feature combinations:
1. Multi-layer hidden states (just the last token)
2. Multi-layer hidden states (averaged across tokens)
3. Only attention patterns
4. Only token probabilities/logits
5. Everything combined

Train on GSM8K, then test on GSM8K/AQuA/MATH to see how well it generalizes.
"""
import os
import numpy as np
import json
import argparse
from datetime import datetime
from tqdm import tqdm
from dataclasses import asdict

from data_loader import load_gsm8k_subset, load_aqua_subset, load_math_subset
from structured_reasoning import StructuredReasoningGenerator, ReasoningStep
from math_prm import MathPRM, OutcomeBasedLabeler
from hidden_state_extractor import HiddenStateExtractor
from probe_trainer import ProbeTrainer

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

# Experiment setups - each one tests a different feature combination
EXPERIMENT_CONFIGS = [
    {
        'name': 'exp1_multi_layer_last_token',
        'description': 'Multi-layer hidden states (last token only)',
        'feature_config': {
            'multi_layer_last_token': True,
            'multi_layer_mean_pooled': False,
            'attention': False,
            'logits': False
        }
    },
    {
        'name': 'exp2_multi_layer_mean_pooled',
        'description': 'Multi-layer hidden states (mean pooled)',
        'feature_config': {
            'multi_layer_last_token': False,
            'multi_layer_mean_pooled': True,
            'attention': False,
            'logits': False
        }
    },
    {
        'name': 'exp3_attention_only',
        'description': 'Attention pattern features only',
        'feature_config': {
            'multi_layer_last_token': False,
            'multi_layer_mean_pooled': False,
            'attention': True,
            'logits': False
        }
    },
    {
        'name': 'exp4_logits_only',
        'description': 'Token logits/probs features only',
        'feature_config': {
            'multi_layer_last_token': False,
            'multi_layer_mean_pooled': False,
            'attention': False,
            'logits': True
        }
    },
    {
        'name': 'exp5_all_features',
        'description': 'Combination of all features',
        'feature_config': {
            'multi_layer_last_token': True,
            'multi_layer_mean_pooled': True,
            'attention': True,
            'logits': True
        }
    }
]


class Config:
    """Experiment configuration."""
    # Training data
    TRAIN_N_SAMPLES = 1024  # GSM8K samples for training
    TRAIN_DATASET = "gsm8k"
    
    # Evaluation data (smaller subsets for speed)
    EVAL_GSM8K_N_SAMPLES = 200
    EVAL_AQUA_N_SAMPLES = 200
    
    # Models
    REASONING_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    PRM_MODEL = "thinkprm"
    
    # Generation settings
    MAX_NEW_TOKENS = 1024
    USE_STRUCTURED_PROMPT = True
    
    # Labeling settings
    PRM_THRESHOLD = 0.5
    USE_OUTCOME_FALLBACK = False  # Disabled to avoid noisy labels on last step
    
    # Debug
    DEBUG_REASONING = False
    DEBUG_PRM = False
    DEBUG_EXTRACTION = False
    
    # Output
    RESULTS_DIR = "results_experiments"


def process_dataset(data, config, reasoning_gen, extractor, prm, outcome_labeler, dataset_name="unknown"):
    """Process a dataset and extract features + labels."""
    all_hidden_states = []
    all_labels = []
    traces_for_review = []
    
    print(f"\n  Processing {len(data)} samples from {dataset_name}...")
    
    # Process in batches to speed things up
    BATCH_SIZE = 32  
    
    for batch_start in tqdm(range(0, len(data), BATCH_SIZE), desc=f"  {dataset_name}"):
        batch_end = min(batch_start + BATCH_SIZE, len(data))
        batch_data = data[batch_start:batch_end]
        
        # Prepare batch
        batch_questions = [item["question"] for item in batch_data]
        batch_expected_answers = [str(item["final_answer"]).strip() for item in batch_data]
        
        try:
            # Generate reasoning for the whole batch at once (much faster)
            batch_traces = reasoning_gen.generate_batch(
                batch_questions,
                max_new_tokens=config.MAX_NEW_TOKENS,
                use_structured_prompt=config.USE_STRUCTURED_PROMPT
            )
            
            # Now process each trace individually (PRM scoring, feature extraction, etc.)
            for i, trace in enumerate(batch_traces):
                question = batch_questions[i]
                expected_answer = batch_expected_answers[i]
                
                try:
                    if len(trace.steps) == 0:
                        continue
                    
                    step_texts = [step.text for step in trace.steps]
                    
                    # Get PRM scores for each step
                    prm_result = prm.score_steps(
                        question=question,
                        steps=step_texts,
                        expected_answer=expected_answer
                    )
                    
                    step_labels = [score.label for score in prm_result.step_scores]
                    step_scores_raw = [score.score for score in prm_result.step_scores]
                    
                    # Check if the final answer is actually correct
                    model_answer = trace.final_answer
                    answer_correct = outcome_labeler._compare_answers(model_answer, expected_answer)
                    
                    if not answer_correct and all(l == 1 for l in step_labels) and config.USE_OUTCOME_FALLBACK:
                        step_labels[-1] = 0
                        step_scores_raw[-1] = 0.0
                    
                    # Extract hidden states
                    step_hidden_states = extractor.extract_step_hidden_states(
                        question, step_texts
                    )
                    
                    # Store results
                    min_len = min(len(step_texts), len(step_hidden_states), len(step_labels))
                    
                    if min_len > 0:
                        all_hidden_states.extend(step_hidden_states[:min_len])
                        all_labels.extend(step_labels[:min_len])
                        
                        # Create detailed trace record (same format as main_v2.py)
                        trace_record = {
                            'sample_index': len(traces_for_review),
                            'dataset': dataset_name,  # Additional field to track dataset
                            'question': question,
                            'expected_answer': expected_answer,
                            'model_answer': model_answer,
                            'answer_correct': answer_correct,
                            'full_reasoning': trace.raw_output,
                            'parsing_issues': trace.parsing_issues,
                            'steps': [],
                            'prm_model': prm_result.prm_model,
                            'prm_overall_score': prm_result.overall_score,
                            'summary': {
                                'total_steps': min_len,
                                'correct_steps': sum(step_labels[:min_len]),
                                'incorrect_steps': min_len - sum(step_labels[:min_len])
                            }
                        }
                        
                        for j in range(min_len):
                            step_info = {
                                'step_number': j + 1,
                                'text': step_texts[j],
                                'prm_score': step_scores_raw[j],
                                'prm_label': step_labels[j],
                                'prm_label_readable': 'correct' if step_labels[j] == 1 else 'incorrect'
                            }
                            trace_record['steps'].append(step_info)
                        
                        traces_for_review.append(trace_record)
                
                except Exception as e:
                    print(f"    Error processing sample {batch_start + i}: {e}")
                    continue
        
        except Exception as e:
            print(f"    Error processing batch {batch_start}-{batch_end}: {e}")
            continue
    
    return all_hidden_states, all_labels, traces_for_review


def extract_features_from_traces(traces, extractor):
    """Extract features from pre-generated traces (fast - forward passes only)."""
    all_hidden_states = []
    all_labels = []
    
    for trace_record in traces:
        question = trace_record['question']
        step_texts = [step['text'] for step in trace_record['steps']]
        step_labels = [step['prm_label'] for step in trace_record['steps']]
        
        if len(step_texts) == 0:
            continue
        
        # Extract features
        step_hidden_states = extractor.extract_step_hidden_states(question, step_texts)
        
        min_len = min(len(step_texts), len(step_hidden_states), len(step_labels))
        if min_len > 0:
            all_hidden_states.extend(step_hidden_states[:min_len])
            all_labels.extend(step_labels[:min_len])
    
    return all_hidden_states, all_labels


def run_experiment_with_traces(exp_config, config, all_traces):
    """Run experiment by extracting features from pre-generated traces."""
    print("\n" + "=" * 80)
    print(f"EXPERIMENT: {exp_config['name']}")
    print(f"Description: {exp_config['description']}")
    print("=" * 80)
    
    # Initialize feature extractor with this experiment's config
    print("\n[1/4] Initializing feature extractor...")
    hidden_extractor = HiddenStateExtractor(
        model_name=config.REASONING_MODEL,
        use_enhanced_features=True,
        feature_config=exp_config['feature_config']
    )
    
    print(f"  Feature dimension: {hidden_extractor.feature_dim}")
    print(f"  Feature config: {exp_config['feature_config']}")
    
    # Extract features from traces
    print("\n[2/4] Extracting features from traces (fast - forward passes only)...")
    
    train_traces = all_traces['GSM8K_train']
    X_train, y_train = extract_features_from_traces(train_traces, hidden_extractor)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    print(f"  Training: {len(X_train)} steps, labels: {np.bincount(y_train, minlength=2)}")
    
    if len(np.unique(y_train)) < 2:
        print("  WARNING: Only one class, skipping")
        return None
    
    # Extract features from eval datasets
    print("\n[3/4] Extracting features from evaluation datasets...")
    eval_results = {}
    
    for eval_name in ['GSM8K', 'AQuA']:
        if eval_name not in all_traces:
            continue
        
        eval_traces = all_traces[eval_name]
        X_eval, y_eval = extract_features_from_traces(eval_traces, hidden_extractor)
        
        if len(X_eval) == 0:
            eval_results[eval_name] = None
            continue
        
        X_eval = np.array(X_eval)
        y_eval = np.array(y_eval)
        
        eval_results[eval_name] = {
            'X': X_eval,
            'y': y_eval,
            'traces': eval_traces
        }
        print(f"  {eval_name}: {len(X_eval)} steps, labels: {np.bincount(y_eval, minlength=2)}")
    
    # Train probe
    print("\n[4/4] Training linear probe...")
    probe_trainer = ProbeTrainer(random_state=42)
    

    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    probe_results = probe_trainer.train(X_train_split, y_train_split, test_size=None)
    probe_results['val_auc'] = None
    probe_results['val_accuracy'] = None
    
    # Evaluate on validation set
    if len(np.unique(y_val)) > 1:
        val_pred = probe_trainer.probe.predict(X_val)
        val_proba = probe_trainer.probe.predict_proba(X_val)[:, 1]
        probe_results['val_auc'] = float(roc_auc_score(y_val, val_proba))
        probe_results['val_accuracy'] = float(accuracy_score(y_val, val_pred))
    
    # Evaluate on all evaluation datasets
    eval_metrics = {}
    for eval_name, eval_data in eval_results.items():
        if eval_data is None:
            continue
        
        X_eval = eval_data['X']
        y_eval = eval_data['y']
        
        if len(np.unique(y_eval)) < 2:
            eval_metrics[eval_name] = {'auc': None, 'accuracy': None}
            continue
        
        eval_pred = probe_trainer.probe.predict(X_eval)
        eval_proba = probe_trainer.probe.predict_proba(X_eval)[:, 1]
        

        eval_metrics[eval_name] = {
            'auc': float(roc_auc_score(y_eval, eval_proba)),
            'accuracy': float(accuracy_score(y_eval, eval_pred)),
            'n_samples': len(X_eval)
        }
    
    # Compile results
    results = {
        'experiment': exp_config['name'],
        'description': exp_config['description'],
        'feature_config': exp_config['feature_config'],
        'feature_dim': int(hidden_extractor.feature_dim),
        'train_stats': {
            'n_samples': len(X_train),
            'n_steps': len(X_train),
            'label_distribution': {
                'incorrect': int(np.sum(y_train == 0)),
                'correct': int(np.sum(y_train == 1))
            }
        },
        'probe_results': {
            'train_auc': float(probe_results['train_auc']),
            'train_accuracy': float(probe_results['train_accuracy']),
            'val_auc': probe_results['val_auc'],
            'val_accuracy': probe_results['val_accuracy']
        },
        'eval_results': eval_metrics
    }
    
    # Save traces for this experiment (same format as main_v2.py)
    traces_file = os.path.join(config.RESULTS_DIR, f"{exp_config['name']}_traces.json")
    all_traces_flat = []
    # Collect all traces from all_traces dict
    for traces in all_traces.values():
        all_traces_flat.extend(traces)
    with open(traces_file, 'w') as f:
        json.dump(all_traces_flat, f, indent=2)
    print(f"  Saved traces: {traces_file} ({len(all_traces_flat)} samples)")
    
    return results


def generate_training_traces(config):
    """Generate reasoning traces for training data (GSM8K) only."""
    print("\n" + "=" * 80)
    print("STEP 1: GENERATE TRAINING REASONING TRACES (GSM8K)")
    print("=" * 80)
    
    # Initialize models
    reasoning_gen = StructuredReasoningGenerator(
        model_name=config.REASONING_MODEL,
        debug=config.DEBUG_REASONING
    )
    
    prm = MathPRM(
        model_name=config.PRM_MODEL,
        debug=config.DEBUG_PRM,
        threshold=config.PRM_THRESHOLD
    )
    
    outcome_labeler = OutcomeBasedLabeler(debug=config.DEBUG_PRM)
    dummy_extractor = HiddenStateExtractor(
        model_name=config.REASONING_MODEL,
        use_enhanced_features=False
    )
    
    # Load training data
    print("\n[1/2] Loading training data...")
    train_data = load_gsm8k_subset(n_samples=config.TRAIN_N_SAMPLES, split="train")
    print(f"  Train (GSM8K): {len(train_data)} samples")
    
    # Generate reasoning traces
    print("\n[2/2] Generating reasoning traces...")
    _, _, train_traces = process_dataset(
        train_data, config, reasoning_gen, dummy_extractor, prm, outcome_labeler, "GSM8K_train"
    )
    print(f"  Generated {len(train_traces)} training traces")
    
    return train_traces, reasoning_gen, prm, outcome_labeler


def generate_eval_traces(config, reasoning_gen, prm, outcome_labeler):
    """Generate reasoning traces for evaluation datasets."""
    print("\n" + "=" * 80)
    print("STEP 3: GENERATE EVALUATION REASONING TRACES")
    print("=" * 80)
    
    dummy_extractor = HiddenStateExtractor(
        model_name=config.REASONING_MODEL,
        use_enhanced_features=False
    )
    
    # Load eval datasets
    print("\n[1/2] Loading evaluation datasets...")
    eval_gsm8k = load_gsm8k_subset(n_samples=config.EVAL_GSM8K_N_SAMPLES, split="test")
    eval_aqua = load_aqua_subset(n_samples=config.EVAL_AQUA_N_SAMPLES, split="test")
    
    print(f"  Eval GSM8K: {len(eval_gsm8k)} samples")
    print(f"  Eval AQuA: {len(eval_aqua)} samples")
    
    # Process eval datasets
    print("\n[2/2] Generating reasoning traces...")
    eval_traces = {}
    
    for eval_name, eval_data in [('GSM8K', eval_gsm8k), ('AQuA', eval_aqua)]:
        print(f"\n  Processing {eval_name}...")
        _, _, traces = process_dataset(
            eval_data, config, reasoning_gen, dummy_extractor, prm, outcome_labeler, eval_name
        )
        eval_traces[eval_name] = traces
        print(f"    Generated {len(traces)} traces")
    
    return eval_traces


def train_probe_for_experiment(exp_config, config, train_traces):
    """Train probe for a specific experiment using training traces."""
    print("\n" + "=" * 80)
    print(f"TRAINING: {exp_config['name']}")
    print(f"Description: {exp_config['description']}")
    print("=" * 80)
    
    # Initialize feature extractor
    print("\n[1/3] Initializing feature extractor...")
    hidden_extractor = HiddenStateExtractor(
        model_name=config.REASONING_MODEL,
        use_enhanced_features=True,
        feature_config=exp_config['feature_config']
    )
    
    print(f"  Feature dimension: {hidden_extractor.feature_dim}")
    print(f"  Feature config: {exp_config['feature_config']}")
    
    # Extract features from training traces
    print("\n[2/3] Extracting features from training traces...")
    X_train, y_train = extract_features_from_traces(train_traces, hidden_extractor)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    print(f"  Training: {len(X_train)} steps, labels: {np.bincount(y_train, minlength=2)}")
    
    if len(np.unique(y_train)) < 2:
        print("  WARNING: Only one class, skipping")
        return None, None, None
    
    # Train probe
    print("\n[3/3] Training linear probe...")
    probe_trainer = ProbeTrainer(random_state=42)

    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    probe_results = probe_trainer.train(X_train_split, y_train_split, test_size=None)
    probe_results['val_auc'] = None
    probe_results['val_accuracy'] = None
    
    if len(np.unique(y_val)) > 1:
        val_pred = probe_trainer.probe.predict(X_val)
        val_proba = probe_trainer.probe.predict_proba(X_val)[:, 1]
        probe_results['val_auc'] = float(roc_auc_score(y_val, val_proba))
        probe_results['val_accuracy'] = float(accuracy_score(y_val, val_pred))
        
        print(f"  Validation AUC: {probe_results['val_auc']:.3f}, Accuracy: {probe_results['val_accuracy']:.3f}")
        print(f"  Validation Classification Report:")
        print(classification_report(y_val, val_pred, target_names=['Incorrect', 'Correct']))
    
    train_stats = {
        'n_samples': len(X_train),
        'n_steps': len(X_train),
        'label_distribution': {
            'incorrect': int(np.sum(y_train == 0)),
            'correct': int(np.sum(y_train == 1))
        }
    }
    
    return probe_trainer, hidden_extractor, {
        'train_stats': train_stats,
        'probe_results': {
            'train_auc': float(probe_results['train_auc']),
            'train_accuracy': float(probe_results['train_accuracy']),
            'val_auc': probe_results['val_auc'],
            'val_accuracy': probe_results['val_accuracy']
        }
    }



def evaluate_probe_for_experiment(exp_config, config, probe_trainer, hidden_extractor, eval_traces):
    """Evaluate trained probe on evaluation datasets."""
    print("\n" + "=" * 80)
    print(f"EVALUATING: {exp_config['name']}")
    print("=" * 80)

    print(f"  Feature dimension: {hidden_extractor.feature_dim}")

    # Extract features and evaluate on each dataset
    eval_metrics = {}

    for eval_name, traces in eval_traces.items():
        print(f"\n  Evaluating on {eval_name}...")
        X_eval, y_eval = extract_features_from_traces(traces, hidden_extractor)
        
        if len(X_eval) == 0:
            eval_metrics[eval_name] = {'auc': None, 'accuracy': None, 'n_samples': 0}
            continue
        
        X_eval = np.array(X_eval)
        y_eval = np.array(y_eval)
        
        print(f"    {len(X_eval)} steps, labels: {np.bincount(y_eval, minlength=2)}")
        
        if len(np.unique(y_eval)) < 2:
            eval_metrics[eval_name] = {'auc': None, 'accuracy': None, 'n_samples': len(X_eval)}
            continue
        
        eval_pred = probe_trainer.probe.predict(X_eval)
        eval_proba = probe_trainer.probe.predict_proba(X_eval)[:, 1]
        
        
        auc = float(roc_auc_score(y_eval, eval_proba))
        acc = float(accuracy_score(y_eval, eval_pred))
        
        eval_metrics[eval_name] = {
            'auc': auc,
            'accuracy': acc,
            'n_samples': len(X_eval)
        }
        print(f"    AUC: {auc:.3f}, Accuracy: {acc:.3f}")
        
        # Print classification report
        print(f"    Classification Report:")
        print(classification_report(y_eval, eval_pred, target_names=['Incorrect', 'Correct']))
    
    return eval_metrics


def main():
    """Run all experiments."""
    parser = argparse.ArgumentParser(description="Run reasoning experiments")
    parser.add_argument("--debug-single-exp", action="store_true", help="Run only the first experiment (last token) for debugging")
    parser.add_argument("--use-existing-traces", action="store_true", help="Use existing reasoning traces from disk instead of generating new ones")
    args = parser.parse_args()

    config = Config()
    
    # Filter experiments if debugging
    active_experiments = EXPERIMENT_CONFIGS
    if args.debug_single_exp:
        print("\n" + "!" * 80)
        print("DEBUG MODE: Running only the first experiment (last token rep)")
        print("!" * 80)
        active_experiments = [EXPERIMENT_CONFIGS[0]]
    
    print("=" * 80)
    print("FEATURE ABLATION EXPERIMENTS")
    print("Train on GSM8K, Evaluate on GSM8K/AQuA")
    print("=" * 80)
    
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # STEP 1: Generate training reasoning traces (GSM8K)
    all_traces_path = os.path.join(config.RESULTS_DIR, "all_reasoning_traces.json")
    if args.use_existing_traces and os.path.exists(all_traces_path):
        print(f"\n[INFO] Loading existing traces from {all_traces_path}...")
        with open(all_traces_path, 'r') as f:
            all_loaded_traces = json.load(f)
        
        train_traces = all_loaded_traces.get('GSM8K_train', [])
        print(f"  Loaded {len(train_traces)} training traces.")
        
        # Initialize models (needed for later steps or if we fall back)
        reasoning_gen = StructuredReasoningGenerator(model_name=config.REASONING_MODEL, debug=config.DEBUG_REASONING)
        prm = MathPRM(model_name=config.PRM_MODEL, debug=config.DEBUG_PRM, threshold=config.PRM_THRESHOLD)
        outcome_labeler = OutcomeBasedLabeler(debug=config.DEBUG_PRM)
    else:
        train_traces, reasoning_gen, prm, outcome_labeler = generate_training_traces(config)
    
    # STEP 2: Train probes for all experiments
    print("\n" + "=" * 80)
    print("STEP 2: TRAINING PROBES FOR ALL EXPERIMENTS")
    print("=" * 80)
    
    trained_probes = {}
    training_results = {}
    
    for exp_config in active_experiments:
        try:
            probe_trainer, hidden_extractor, train_result = train_probe_for_experiment(exp_config, config, train_traces)
            if probe_trainer is not None:
                trained_probes[exp_config['name']] = {
                    'probe_trainer': probe_trainer,
                    'extractor': hidden_extractor,  # Keep extractor for evaluation
                    'exp_config': exp_config,
                    'feature_dim': hidden_extractor.feature_dim
                }
                training_results[exp_config['name']] = train_result
                print(f"\n  ✓ Trained: {exp_config['name']}")
        
        except Exception as e:
            print(f"\n  ✗ ERROR training {exp_config['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # STEP 3: Generate evaluation reasoning traces
    if args.use_existing_traces and os.path.exists(all_traces_path):
        print(f"\n[INFO] Using existing evaluation traces from {all_traces_path}...")
        eval_traces = {}
        for key in ['GSM8K', 'AQuA']:
            if key in all_loaded_traces:
                eval_traces[key] = all_loaded_traces[key]
                print(f"  Loaded {len(eval_traces[key])} traces for {key}")
    else:
        eval_traces = generate_eval_traces(config, reasoning_gen, prm, outcome_labeler)
    
    # Save all traces (only if we generated new ones to avoid re-saving same data, but re-saving is fine)
    if not args.use_existing_traces:
        all_traces = {'GSM8K_train': train_traces, **eval_traces}
        traces_file = os.path.join(config.RESULTS_DIR, "all_reasoning_traces.json")
        with open(traces_file, 'w') as f:
            json.dump(all_traces, f, indent=2)
        print(f"\n  Saved all traces: {traces_file}")
    
    # STEP 4: Evaluate all trained probes on eval datasets
    print("\n" + "=" * 80)
    print("STEP 4: EVALUATING PROBES ON EVALUATION DATASETS")
    print("=" * 80)
    
    all_results = []
    
    for exp_name, probe_info in trained_probes.items():
        try:
            exp_config = probe_info['exp_config']
            probe_trainer = probe_info['probe_trainer']
            hidden_extractor = probe_info['extractor']
            
            eval_metrics = evaluate_probe_for_experiment(exp_config, config, probe_trainer, hidden_extractor, eval_traces)
            
            # Compile full results
            train_result = training_results[exp_name]
            results = {
                'experiment': exp_name,
                'description': exp_config['description'],
                'feature_config': exp_config['feature_config'],
                'feature_dim': probe_info['feature_dim'],
                **train_result,
                'eval_results': eval_metrics
            }
            
            all_results.append(results)
            
            # Save individual experiment result
            exp_file = os.path.join(config.RESULTS_DIR, f"{exp_name}_results.json")
            with open(exp_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n  ✓ Saved: {exp_file}")
        
        except Exception as e:
            print(f"\n  ✗ ERROR evaluating {exp_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'train_samples': config.TRAIN_N_SAMPLES,
            'eval_samples': {
                'gsm8k': config.EVAL_GSM8K_N_SAMPLES,
                'aqua': config.EVAL_AQUA_N_SAMPLES
            },
            'model': config.REASONING_MODEL,
            'prm': config.PRM_MODEL
        },
        'experiments': all_results
    }
    
    summary_file = os.path.join(config.RESULTS_DIR, "all_experiments_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {config.RESULTS_DIR}/")
    print(f"  - Individual results: exp*_results.json")
    print(f"  - Summary: all_experiments_summary.json")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Experiment':<30} {'Feature Dim':<12} {'Train AUC':<10} {'GSM8K AUC':<10} {'AQuA AUC':<10}")
    print("-" * 80)
    
    for result in all_results:
        exp_name = result['experiment']
        feat_dim = result['feature_dim']
        train_auc = result['probe_results']['train_auc']
        gsm8k_auc = result['eval_results'].get('GSM8K', {}).get('auc', 'N/A')
        aqua_auc = result['eval_results'].get('AQuA', {}).get('auc', 'N/A')
        
        print(f"{exp_name:<30} {feat_dim:<12} {train_auc:<10.3f} {gsm8k_auc:<10.3f} {aqua_auc:<10.3f}")


if __name__ == "__main__":
    main()

