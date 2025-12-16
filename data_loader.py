"""
Load datasets for the experiments.

We use GSM8K for training, and test on GSM8K, AQuA, and MATH to see how well things generalize.
"""
import re
from datasets import load_dataset
from typing import List, Dict, Tuple, Optional


def load_gsm8k_subset(n_samples: int = 100, split: str = "test") -> List[Dict]:
    """
    Load a subset of GSM8K dataset.
    
    Args:
        n_samples: Number of samples to load
        split: Dataset split to use ('train' or 'test')
    
    Returns:
        List of dictionaries with 'question' and 'answer' keys
    """
    print(f"  Loading GSM8K ({split})...")
    dataset = load_dataset("gsm8k", "main")
    data = dataset[split]
    
    # Take a subset
    subset = data.select(range(min(n_samples, len(data))))
    
    # Parse questions and answers
    parsed_data = []
    for item in subset:
        question = item["question"]
        answer = item["answer"]
        
        # Extract the final numerical answer
        # GSM8K answers are in format: "step1\nstep2\n...\n#### 123"
        answer_match = re.search(r"####\s*([-+]?\d*\.?\d+)", answer)
        final_answer = answer_match.group(1) if answer_match else None
        
        parsed_data.append({
            "question": question,
            "answer": answer,
            "final_answer": final_answer,
            "dataset": "GSM8K"
        })
    
    return parsed_data


def parse_reasoning_steps(answer_text: str) -> List[str]:
    """
    Parse reasoning steps from answer text.
    
    Args:
        answer_text: The answer text containing reasoning steps
    
    Returns:
        List of reasoning step strings
    """
    # Split by newlines and filter out the final answer line
    lines = answer_text.strip().split("\n")
    steps = [line.strip() for line in lines if not line.strip().startswith("####")]
    return steps


def extract_final_answer(answer_text: str) -> str:
    """
    Extract the final numerical answer from answer text.
    
    Args:
        answer_text: The answer text
    
    Returns:
        Final answer as string
    """
    match = re.search(r"####\s*([-+]?\d*\.?\d+)", answer_text)
    return match.group(1) if match else None


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract answer from \boxed{...} allowing for nested braces.
    """
    if "\\boxed{" not in text:
        return None
        
    # Find all start indices
    start_indices = [m.start() for m in re.finditer(r"\\boxed\{", text)]
    
    # Try the last one first as it usually contains the final answer
    for start_idx in reversed(start_indices):
        idx = start_idx + 7  # Length of "\boxed{"
        depth = 1
        content = ""
        
        for i in range(idx, len(text)):
            char = text[i]
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
            
            if depth == 0:
                return content
            content += char
            
    return None


def load_aqua_subset(n_samples: int = 100, split: str = "test") -> List[Dict]:
    """
    Load a subset of AQuA dataset.
    
    Uses: hails/agieval-aqua-rat
    Format: 
    - query (question)
    - options/choices (list of answer choices)
    - rational (step-by-step reasoning/explanation)
    - label/gold (correct answer index or letter)
    
    Args:
        n_samples: Number of samples to load
        split: Dataset split to use ('train' or 'test')
    
    Returns:
        List of dictionaries with 'question', 'answer' (rationale), and 'final_answer' keys
    """
    print(f"  Loading AQuA ({split})...")
    try:
        # Load AQuA dataset from hails/agieval-aqua-rat
        dataset = load_dataset("hails/agieval-aqua-rat", split=split)
    except Exception as e:
        print(f"  Warning: Could not load AQuA dataset (hails/agieval-aqua-rat) with split '{split}': {e}")
        try:
            # Try 'train' if 'test' fails (or vice versa if needed)
            alt_split = 'train' if split == 'test' else 'test'
            print(f"  Trying split '{alt_split}' instead...")
            dataset = load_dataset("hails/agieval-aqua-rat", split=alt_split)
        except Exception as e2:
            print(f"  Failed to load AQuA dataset: {e2}")
            print("  Using GSM8K as fallback (WARNING: Data duplication)")
            return load_gsm8k_subset(n_samples, split)
    
    # Take a subset
    subset = dataset.select(range(min(n_samples, len(dataset))))
    
    parsed_data = []
    for item in subset:
        # AQuA format: query (question), options (list of answer choices), 
        # rational (step-by-step reasoning), label (correct index)
        question = item.get("query", item.get("question", ""))
        options = item.get("options", item.get("choices", []))
        rational = item.get("rational", item.get("rationale", item.get("explanation", "")))
        label = item.get("label", item.get("correct", item.get("gold", None)))
        
        # Append options to question to make it self-contained
        if options:
            question += "\n\nOptions:"
            for i, opt in enumerate(options):
                # Ensure option has letter prefix if missing
                prefix = f"{chr(65+i)})"
                if not opt.strip().startswith(prefix) and not re.match(r'^[A-E]\)', opt.strip()):
                     question += f"\n{prefix} {opt}"
                else:
                     question += f"\n{opt}"

        # Extract final answer
        final_answer = None
        if label is not None:
            # Normalize label to index
            label_idx = -1
            
            # Handle list case (e.g. gold=[0])
            if isinstance(label, list) and len(label) > 0:
                 label = label[0]
                 
            if isinstance(label, int):
                label_idx = label
            elif isinstance(label, str):
                label = label.strip().upper()
                if label.isdigit():
                    label_idx = int(label)
                elif len(label) == 1 and 'A' <= label <= 'E':
                    label_idx = ord(label) - ord('A')

            # Extract text using index
            if options and 0 <= label_idx < len(options):
                answer_text = options[label_idx]
                # Try to extract number from answer string like "(A)45 Min" -> "45"
                # Look for numbers, possibly decimal
                # AQuA options often look like: "A) 24", "B) 32", etc.
                # Remove the letter prefix first
                clean_text = re.sub(r'^[A-E]\)?\s*', '', answer_text)
                
                # Now extract the number
                num_match = re.search(r'([-+]?[0-9,]*\.?[0-9]+)', clean_text)
                if num_match:
                    final_answer = num_match.group(1).replace(',', '')
                else:
                    final_answer = clean_text # Fallback to text if no number
            else:
                # Fallback: just return the label itself if we can't map it
                final_answer = str(label)
        
        parsed_data.append({
            "question": question,
            "answer": rational if rational else question,  # Use rational (reasoning) if available
            "final_answer": final_answer,
            "dataset": "AQuA"
        })
    
    return parsed_data

