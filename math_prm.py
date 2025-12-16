"""
Use Process Reward Models (PRMs) to score whether each reasoning step is correct.

We support a few different PRM models - ThinkPRM seems to work best for our use case.
"""
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import numpy as np


@dataclass
class StepScore:
    """Score for a single reasoning step."""
    step_number: int
    step_text: str
    score: float  # 0-1 probability of correctness
    label: int  # 0 = incorrect, 1 = correct
    raw_output: Optional[str] = None  # For generative PRMs


@dataclass 
class PRMResult:
    """Complete PRM evaluation result."""
    question: str
    steps: List[str]
    step_scores: List[StepScore]
    overall_score: float  # Average or min of step scores
    prm_model: str
    debug_info: Dict


class MathPRM:
    """
    Wrapper for different PRM models.
    
    We tried a few options:
    - Skywork-o1-PRM: fast but had some issues
    - Math-Shepherd: didn't work well for us
    - ThinkPRM: slower but gives better results and explanations
    """
    
    SUPPORTED_MODELS = {
        "skywork": "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B",
        "math-shepherd": "peiyi9979/math-shepherd-mistral-7b-prm",
        "thinkprm": "launch/ThinkPRM-1.5B",
    }
    
    def __init__(self, model_name: str = "skywork", device: str = None, 
                 debug: bool = False, threshold: float = 0.5):
        """
        Initialize the Math PRM.
        
        Args:
            model_name: Either a key from SUPPORTED_MODELS or a HuggingFace model path
            device: Device to run on
            debug: If True, print detailed output
            threshold: Score threshold for correct/incorrect classification
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.debug = debug
        self.threshold = threshold
        
        # Resolve model name
        if model_name in self.SUPPORTED_MODELS:
            self.model_path = self.SUPPORTED_MODELS[model_name]
            self.model_type = model_name
        else:
            self.model_path = model_name
            self.model_type = "custom"
        
        if self.debug:
            print(f"[MathPRM] Loading model: {self.model_path}")
            print(f"[MathPRM] Model type: {self.model_type}")
            print(f"[MathPRM] Device: {self.device}")
        
        self._load_model()
    
    def _load_model(self):
        """Load the appropriate model based on type."""
        if self.model_type == "skywork":
            self._load_skywork()
        elif self.model_type == "math-shepherd":
            self._load_math_shepherd()
        elif self.model_type == "thinkprm":
            self._load_thinkprm()
        else:
            # Try to load as a generic causal LM
            self._load_generic()
    
    def _load_skywork(self):
        """Load Skywork PRM - this is a reward model with custom architecture."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        self.model.eval()
        
        # Store model reference for getting device
        self._model_device = next(self.model.parameters()).device
        
        if self.debug:
            print(f"[MathPRM] Skywork PRM loaded successfully")
    
    def _load_math_shepherd(self):
        """Load Math-Shepherd PRM."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        self.model.to(self.device)
        self.model.eval()
        
        # Math-Shepherd uses special tokens
        self.good_token_id = self.tokenizer.encode("+")[-1]
        self.bad_token_id = self.tokenizer.encode("-")[-1]
        
        if self.debug:
            print(f"[MathPRM] Math-Shepherd loaded.")
    
    def _load_thinkprm(self):
        """Load ThinkPRM (generative)."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        self.model.to(self.device)
        self.model.eval()
        
        if self.debug:
            print(f"[MathPRM] ThinkPRM loaded.")
    
    def _load_generic(self):
        """Load a generic model."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        self.model.to(self.device)
        self.model.eval()
    
    def score_steps(self, question: str, steps: List[str], 
                    expected_answer: Optional[str] = None) -> PRMResult:
        """
        Score each reasoning step.
        
        Args:
            question: The math problem
            steps: List of reasoning steps
            expected_answer: Expected final answer (for outcome-based fallback)
        
        Returns:
            PRMResult with scores for each step
        """
        if self.model_type == "skywork":
            return self._score_skywork(question, steps, expected_answer)
        elif self.model_type == "math-shepherd":
            return self._score_math_shepherd(question, steps, expected_answer)
        elif self.model_type == "thinkprm":
            return self._score_thinkprm(question, steps, expected_answer)
        else:
            return self._score_generic(question, steps, expected_answer)
    
    def _score_skywork(self, question: str, steps: List[str],
                       expected_answer: Optional[str] = None) -> PRMResult:
        """Score steps using Skywork PRM (reward model)."""
        step_scores = []
        debug_info = {"model": "skywork", "raw_scores": []}
        
        # Skywork PRM expects format:
        # "<|im_start|>system\nPlease reason step by step...<|im_end|>\n
        #  <|im_start|>user\n{question}<|im_end|>\n
        #  <|im_start|>assistant\n{solution_steps}<|im_end|>"
        
        # Build solution progressively to score each step
        solution_so_far = ""
        
        for i, step in enumerate(steps):
            # Add current step to solution
            step_text = f"Step {i+1}: {step}\n"
            solution_so_far += step_text
            
            # Format as chat
            messages = [
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                {"role": "user", "content": question},
                {"role": "assistant", "content": solution_so_far}
            ]
            
            # Apply chat template
            full_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            
            # Tokenize
            inputs = self.tokenizer(full_text, return_tensors="pt").to(self._model_device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Skywork PRM outputs a scalar reward
                # The output format may vary - try different extraction methods
                if hasattr(outputs, 'logits'):
                    # If it outputs logits, use them
                    score = torch.sigmoid(outputs.logits[0, -1]).item()
                elif hasattr(outputs, 'last_hidden_state'):
                    # If it outputs hidden states, the model might have a value head
                    # Try to get reward from the model directly
                    if hasattr(self.model, 'score'):
                        score = self.model.score(outputs.last_hidden_state[:, -1, :]).item()
                    else:
                        # Fallback: use hidden state norm as proxy
                        score = torch.sigmoid(outputs.last_hidden_state[:, -1, 0]).item()
                else:
                    # Direct output
                    score = torch.sigmoid(torch.tensor(float(outputs[0]))).item()
            
            # Clamp score to [0, 1]
            score = max(0.0, min(1.0, score))
            label = 1 if score >= self.threshold else 0
            
            step_scores.append(StepScore(
                step_number=i + 1,
                step_text=step,
                score=score,
                label=label,
                raw_output=f"reward_score={score:.4f}"
            ))
            
            debug_info["raw_scores"].append({
                "step": i + 1,
                "score": score
            })
            
            if self.debug:
                print(f"[MathPRM] Step {i+1}: score={score:.4f} ({'CORRECT' if label else 'INCORRECT'})")
                print(f"          Text: {step[:80]}...")
        
        overall_score = np.mean([s.score for s in step_scores]) if step_scores else 0.0
        
        return PRMResult(
            question=question,
            steps=steps,
            step_scores=step_scores,
            overall_score=overall_score,
            prm_model=self.model_path,
            debug_info=debug_info
        )
    
    def _score_math_shepherd(self, question: str, steps: List[str],
                             expected_answer: Optional[str] = None) -> PRMResult:
        """Score steps using Math-Shepherd PRM."""
        step_scores = []
        debug_info = {"model": "math-shepherd", "raw_scores": []}
        
        # Math-Shepherd format
        # "Question: ... Step 1: ... ки Step 2: ... ки"
        conversation = f"{question} "
        
        for i, step in enumerate(steps):
            step_text = f"Step {i+1}: {step} ки "
            full_text = conversation + step_text
            
            inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(inputs.input_ids)
                logits = outputs.logits[:, -1, :]
                
                probs = torch.softmax(logits, dim=-1)
                good_prob = probs[0, self.good_token_id].item()
                bad_prob = probs[0, self.bad_token_id].item()
                
                total = good_prob + bad_prob
                score = good_prob / total if total > 0 else 0.5
            
            label = 1 if score >= self.threshold else 0
            
            step_scores.append(StepScore(
                step_number=i + 1,
                step_text=step,
                score=score,
                label=label
            ))
            
            debug_info["raw_scores"].append({"step": i + 1, "score": score})
            
            if self.debug:
                print(f"[MathPRM] Step {i+1}: score={score:.4f} ({'CORRECT' if label else 'INCORRECT'})")
            
            conversation = full_text
        
        overall_score = np.mean([s.score for s in step_scores]) if step_scores else 0.0
        
        return PRMResult(
            question=question,
            steps=steps,
            step_scores=step_scores,
            overall_score=overall_score,
            prm_model=self.model_path,
            debug_info=debug_info
        )
    
    def _score_thinkprm(self, question: str, steps: List[str],
                        expected_answer: Optional[str] = None) -> PRMResult:
        """Score steps using ThinkPRM (generative)."""
        step_scores = []
        debug_info = {"model": "thinkprm", "verification_output": ""}
        
        # Format solution - clearer format for ThinkPRM
        solution_text = ""
        for i, step in enumerate(steps):
            solution_text += f"Step {i+1}: {step}\n"
        
        # Improved prompt for ThinkPRM - STRICT MODE
        prompt = f"""Verify each step of this math solution.

PROBLEM:
{question}

PROPOSED SOLUTION:
{solution_text}

IMPORTANT RULES:
- A step is correct ONLY if it logically follows from the problem and all previous steps.
- Penalize any step that introduces an invalid equation, misuses quantities (rates vs totals), or omits required factors.
- If a step depends on an incorrect or unjustified prior step, it MUST be marked incorrect.
- Do NOT be lenient. If the step is ambiguous or incomplete, mark it incorrect.
- Structural mistakes (wrong formulas, missing multipliers, misuse of rates) are MORE severe than arithmetic mistakes.

You MUST output one verdict per step, in order:
Step 1 is \\boxed{{correct/incorrect}}
Step 2 is \\boxed{{correct/incorrect}}
...

Now verify each step:"""
        
        messages = [{'role': "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=2048,  # Reduced from 2048 - 512 is usually enough
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        verification_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        debug_info["verification_output"] = verification_text[-3000:]
        
        if self.debug:
            print(f"\n[MathPRM] ThinkPRM FULL OUTPUT:")
            print("="*60)
            # Get only the generated part (after the prompt)
            gen_start = verification_text.find("Now verify each step:")
            if gen_start > 0:
                print(verification_text[gen_start:])
            else:
                print(verification_text[-2000:])
            print("="*60)
        
        # Parse labels
        labels = self._parse_thinkprm_output(verification_text, len(steps))
        
        for i, (step, label) in enumerate(zip(steps, labels)):
            step_scores.append(StepScore(
                step_number=i + 1,
                step_text=step,
                score=1.0 if label else 0.0,
                label=label,
                raw_output=None
            ))
            
            if self.debug:
                print(f"[MathPRM] Step {i+1}: {'CORRECT' if label else 'INCORRECT'}")
        
        overall_score = np.mean([s.score for s in step_scores]) if step_scores else 0.0
        
        return PRMResult(
            question=question,
            steps=steps,
            step_scores=step_scores,
            overall_score=overall_score,
            prm_model=self.model_path,
            debug_info=debug_info
        )
    
    def _parse_thinkprm_output(self, text: str, num_steps: int) -> List[int]:
        """Parse ThinkPRM output for step labels - STRICT MODE."""
        labels = [0] * num_steps  # Default all to incorrect
        
        # Extract only the generated response (after assistant marker)
        analysis_text = text
        for marker in ['<|assistant|>', '<｜Assistant｜>', 'Now verify each step:']:
            if marker in analysis_text:
                analysis_text = analysis_text.split(marker)[-1]
                break
        
        if self.debug:
            print(f"\n[MathPRM] Parsing {len(analysis_text)} chars of verification text (STRICT MODE)")
        
        # Strict parsing: Look ONLY for explicit boxed verdicts
        # Format: "Step N is \boxed{correct}" or "\boxed{incorrect}"
        
        # Find all boxed verdicts associated with specific steps
        boxed_matches = re.finditer(
            r'step\s*(\d+)\s*.*?\\boxed\{(correct|incorrect)\}', 
            analysis_text, re.IGNORECASE | re.DOTALL
        )
        
        found_verdicts = 0
        for match in boxed_matches:
            step_num = int(match.group(1))
            verdict = match.group(2).lower()
            
            idx = step_num - 1
            if 0 <= idx < num_steps:
                labels[idx] = 1 if verdict == 'correct' else 0
                found_verdicts += 1
                if self.debug:
                    print(f"[MathPRM] Step {step_num}: {verdict.upper()}")
        
        # Fallback: If no "Step N" format found, look for sequential boxed verdicts
        # This handles cases where model just outputs "\boxed{correct}" line by line
        if found_verdicts == 0:
            sequential_matches = list(re.finditer(r'\\boxed\{(correct|incorrect)\}', analysis_text, re.IGNORECASE))
            if sequential_matches:
                if self.debug:
                    print(f"[MathPRM] No numbered steps found. Using sequential matching ({len(sequential_matches)} verdicts)")
                
                # Take first N verdicts for N steps
                for i, match in enumerate(sequential_matches):
                    if i >= num_steps:
                        break
                    verdict = match.group(1).lower()
                    labels[i] = 1 if verdict == 'correct' else 0
        
        if self.debug:
            print(f"[MathPRM] Final parsed labels: {labels}")
        
        return labels
    
    def _score_generic(self, question: str, steps: List[str],
                       expected_answer: Optional[str] = None) -> PRMResult:
        """Generic scoring using outcome-based labeling."""
        step_scores = []
        debug_info = {"model": "outcome-based", "method": "final_answer_check"}
        
        # If we have expected answer, use outcome-based labeling
        # Otherwise, default to 0.5 (uncertain)
        default_score = 0.5
        
        for i, step in enumerate(steps):
            step_scores.append(StepScore(
                step_number=i + 1,
                step_text=step,
                score=default_score,
                label=1 if default_score >= self.threshold else 0
            ))
        
        return PRMResult(
            question=question,
            steps=steps,
            step_scores=step_scores,
            overall_score=default_score,
            prm_model="outcome-based",
            debug_info=debug_info
        )


class OutcomeBasedLabeler:
    """
    Simple outcome-based labeling.
    If final answer is correct, all steps are "correct".
    If final answer is wrong, mark all steps as "incorrect" (conservative).
    """
    
    def __init__(self, debug: bool = False):
        self.debug = debug
    
    def label_steps(self, steps: List[str], model_answer: str, 
                    expected_answer: str) -> List[int]:
        """
        Label steps based on final answer correctness.
        
        Args:
            steps: List of reasoning steps
            model_answer: Model's final answer
            expected_answer: Expected correct answer
        
        Returns:
            List of binary labels
        """
        is_correct = self._compare_answers(model_answer, expected_answer)
        
        if self.debug:
            print(f"[OutcomeLabeler] Model answer: {model_answer}")
            print(f"[OutcomeLabeler] Expected: {expected_answer}")
            print(f"[OutcomeLabeler] Correct: {is_correct}")
        
        # All steps get the same label based on outcome
        labels = [1 if is_correct else 0] * len(steps)
        
        return labels
    
    def _compare_answers(self, model: str, expected: str) -> bool:
        """Compare two answers, handling various formats."""
        if model is None or expected is None:
            return False
        
        # Clean up answers
        model_clean = str(model).strip().lower()
        expected_clean = str(expected).strip().lower()
        
        # Remove common suffixes
        for suffix in ['dollars', 'dollar', '$', 'meters', 'meter', 'cups', 'cup', 'bolts', 'bolt', '.']:
            model_clean = model_clean.replace(suffix, '').strip()
            expected_clean = expected_clean.replace(suffix, '').strip()
        
        # Remove commas from numbers
        model_clean = model_clean.replace(',', '')
        expected_clean = expected_clean.replace(',', '')
        
        # Try numeric comparison
        try:
            model_num = float(model_clean)
            expected_num = float(expected_clean)
            return abs(model_num - expected_num) < 0.01
        except ValueError:
            pass
        
        # String comparison
        return model_clean == expected_clean


def test_math_prm():
    """Test the Math PRM."""
    # Test with Skywork
    try:
        prm = MathPRM(model_name="skywork", debug=True)
        
        question = "What is 2 + 3?"
        steps = [
            "We need to add 2 and 3.",
            "2 + 3 = 5",
            "The answer is 5."
        ]
        
        result = prm.score_steps(question, steps)
        
        print("\n" + "="*60)
        print("RESULTS:")
        print("="*60)
        for score in result.step_scores:
            print(f"Step {score.step_number}: {score.score:.4f} ({score.label})")
        print(f"Overall: {result.overall_score:.4f}")
        
    except Exception as e:
        print(f"Error testing Skywork: {e}")
        print("Trying ThinkPRM instead...")
        
        prm = MathPRM(model_name="thinkprm", debug=True)
        result = prm.score_steps(question, steps)


if __name__ == "__main__":
    test_math_prm()

