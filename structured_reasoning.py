"""
Generate step-by-step reasoning from the model.

The goal is to get the model to actually show its work, not just describe what it's doing.
We use a specific prompt format to encourage this.
"""
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class ReasoningStep:
    """A single reasoning step."""
    step_number: int
    text: str  # The full step content


@dataclass
class ReasoningTrace:
    """Complete reasoning trace with metadata."""
    question: str
    raw_output: str  # Full model output
    steps: List[ReasoningStep]
    final_answer: Optional[str]
    parsing_issues: List[str]  # Any issues during parsing


class StructuredReasoningGenerator:
    """Generate structured step-by-step reasoning with explicit calculations."""
    
    # This prompt seems to work well for getting clear step-by-step reasoning
    STRUCTURED_PROMPT = """You are a precise math reasoning engine. Solve the problem below by breaking it down into numbered steps.

Rules:
1. Format each step as "Step N: [reasoning]".
2. Each step should contain one logical deduction or calculation.
3. At the very end, output the final numerical answer strictly in the format "FINAL_ANSWER: X".
4. Do NOT use units, text, or trailing punctuation in the final answer line.

Example:
Problem: If a shop sells 50 apples a day, how many does it sell in a week?
Step 1: A week has 7 days.
Step 2: We multiply the daily sales by the number of days: 50 * 7 = 350.
FINAL_ANSWER: 350

Problem: {question}

Solution:
"""

    # Simpler prompt if the structured one doesn't work well
    SIMPLE_PROMPT = """Solve this step by step. Show your work with actual numbers.

Question: {question}

Let me solve this step by step:

Step 1:"""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", 
                 device: str = None, debug: bool = False):
        """
        Initialize the structured reasoning generator.
        
        Args:
            model_name: HuggingFace model name
            device: Device to run on
            debug: If True, print detailed output
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.debug = debug
        
        if self.debug:
            print(f"[StructuredReasoning] Loading model: {model_name}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.model.to(self.device)
        self.model.eval()
        
        # Try to compile the model for faster inference (only works with PyTorch 2.0+)
        try:
            if hasattr(torch, 'compile'):
                self.model = torch.compile(self.model, mode="reduce-overhead")
                if self.debug:
                    print(f"[StructuredReasoning] Model compiled with torch.compile")
        except Exception as e:
            if self.debug:
                print(f"[StructuredReasoning] torch.compile not available: {e}")
        
        # Some tokenizers don't have a pad token, so we use eos token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Check if this is an instruct/chat model (they need special formatting)
        self.is_instruct = "instruct" in model_name.lower() or "chat" in model_name.lower()
        if self.debug:
            print(f"[StructuredReasoning] Detected instruct model: {self.is_instruct}")
    
    def generate(self, question: str, max_new_tokens: int = 1024,
                 use_structured_prompt: bool = True) -> ReasoningTrace:
        """Generate reasoning for a single question (just wraps the batch version)."""
        return self.generate_batch([question], max_new_tokens, use_structured_prompt)[0]
    
    def generate_batch(self, questions: List[str], max_new_tokens: int = 1024,
                       use_structured_prompt: bool = True) -> List[ReasoningTrace]:
        """
        Generate structured reasoning traces for multiple questions (batch processing).
        
        Args:
            questions: List of math problems
            max_new_tokens: Maximum tokens to generate
            use_structured_prompt: Whether to use the structured format prompt
        
        Returns:
            List of ReasoningTrace objects with parsed steps
        """
        # Build prompts for all questions in the batch
        formatted_prompts = []
        for question in questions:
            if use_structured_prompt:
                prompt_text = self.STRUCTURED_PROMPT.format(question=question)
            else:
                prompt_text = self.SIMPLE_PROMPT.format(question=question)
            
            # Instruct models need special formatting with chat templates
            if self.is_instruct:
                messages = [{"role": "user", "content": prompt_text}]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                formatted_prompt = prompt_text
            
            formatted_prompts.append(formatted_prompt)
        
        # Tokenize everything at once with padding so we can batch process
        inputs = self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512  # This should be enough for most questions
        ).to(self.device)
        
        if self.debug:
            print(f"\n[StructuredReasoning] Batch size: {len(questions)}")
        
        # Generate for all prompts in the batch at once
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy is faster and more deterministic
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,  # KV cache speeds things up
            )
        
        # Now decode and parse each output
        traces = []
        for i, question in enumerate(questions):
            # Decode this specific output
            full_output = self.tokenizer.decode(outputs[i], skip_special_tokens=True)
            
            # Need to extract just the generated part (not the prompt)
            formatted_prompt = formatted_prompts[i]
            if self.is_instruct:
                # Instruct models usually have special markers
                if "<|assistant|>" in full_output:
                    raw_output = full_output.split("<|assistant|>")[-1].strip()
                elif "<｜Assistant｜>" in full_output:
                    raw_output = full_output.split("<｜Assistant｜>")[-1].strip()
                else:
                    # Fallback: try to find where the solution starts
                    raw_output = full_output.split("Solution:")[-1].strip() if "Solution:" in full_output else full_output
            else:
                # For non-instruct models, just remove the prompt
                raw_output = full_output[len(formatted_prompt):].strip()
            
            if self.debug and i == 0:
                print(f"\n{'='*60}")
                print("[StructuredReasoning] RAW MODEL OUTPUT (first):")
                print('='*60)
                print(raw_output)
                print('='*60)
            
            # Parse the output into structured steps
            trace = self._parse_output(question, raw_output, use_structured_prompt)
            traces.append(trace)
        
        if self.debug:
            print(f"\n[StructuredReasoning] Batch complete: {len(traces)} traces generated")
            for i, trace in enumerate(traces[:3]):  # Show first 3
                print(f"  Trace {i+1}: {len(trace.steps)} steps, answer: {trace.final_answer}")
        
        return traces
    
    def _parse_output(self, question: str, raw_output: str, 
                      structured: bool) -> ReasoningTrace:
        """Parse the model output into structured steps."""
        steps = []
        parsing_issues = []
        final_answer = None
        
        # Clean the output first - remove common noise
        cleaned_output = self._clean_output(raw_output)
        
        # Try multiple parsing strategies in order of preference
        
        # Strategy 1: Parse <step>...</step> blocks (with or without Action/Calculation/Result)
        step_blocks = self._parse_step_tags(cleaned_output)
        if step_blocks:
            steps = step_blocks
            parsing_issues.append("Parsed using <step> tags")
        
        # Strategy 2: Parse markdown "### Step N:" or "Step N:" format
        if not steps:
            markdown_steps = self._parse_markdown_steps(cleaned_output)
            if markdown_steps:
                steps = markdown_steps
                parsing_issues.append("Parsed using Step N: format")
        
        # Strategy 3: Parse numbered list format "1. ", "2. "
        if not steps:
            numbered_steps = self._parse_numbered_list(cleaned_output)
            if numbered_steps:
                steps = numbered_steps
                parsing_issues.append("Parsed using numbered list format")
        
        # Strategy 4: Fallback - split by double newlines for logical paragraphs
        if not steps:
            paragraph_steps = self._parse_paragraphs(cleaned_output)
            if paragraph_steps:
                steps = paragraph_steps
                parsing_issues.append("Parsed using paragraph splitting")
        
        # Extract final answer
        answer_match = re.search(r'<answer>\s*(.+?)\s*</answer>', raw_output, re.IGNORECASE)
        if answer_match:
            final_answer = answer_match.group(1).strip()
        else:
            final_answer = self._extract_final_answer(cleaned_output)
            if final_answer:
                parsing_issues.append("No <answer> tag, extracted from text")
        
        return ReasoningTrace(
            question=question,
            raw_output=raw_output,
            steps=steps,
            final_answer=final_answer,
            parsing_issues=parsing_issues
        )
    
    def _clean_output(self, text: str) -> str:
        """Clean the raw output by removing noise."""
        # Remove common noise patterns
        noise_patterns = [
            r'^assistant\s*',  # "assistant" at start
            r'^\s*<\|.*?\|>\s*',  # Special tokens like <|assistant|>
            r'^\s*$',  # Empty lines at start
        ]
        
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            cleaned = line
            # Skip lines that are just "assistant" or empty
            if cleaned.strip().lower() in ['assistant', '']:
                continue
            # Skip lines that are just tags like "<step>" or "</step>"
            if re.match(r'^\s*</?step>\s*$', cleaned, re.IGNORECASE):
                continue
            cleaned_lines.append(cleaned)
        
        return '\n'.join(cleaned_lines)
    
    def _parse_step_tags(self, text: str) -> List[ReasoningStep]:
        """Parse <step>...</step> blocks."""
        steps = []
        
        # Find all <step>...</step> blocks (flexible - any content inside)
        step_block_pattern = re.compile(r'<step>\s*(.*?)\s*</step>', re.DOTALL | re.IGNORECASE)
        
        matches = list(step_block_pattern.finditer(text))
        
        if matches:
            for i, match in enumerate(matches, 1):
                block_content = match.group(1).strip()
                
                if not block_content:
                    continue
                
                # Clean up the content
                cleaned = self._clean_step_content(block_content)
                
                if cleaned:
                    steps.append(ReasoningStep(
                        step_number=i,
                        text=cleaned
                    ))
        
        return steps
    
    def _parse_markdown_steps(self, text: str) -> List[ReasoningStep]:
        """Parse markdown Step N: format."""
        steps = []
        
        # Pattern for "### Step N:" or "Step N:" or "**Step N:**"
        step_pattern = re.compile(
            r'(?:^|\n)\s*(?:#{1,3}\s*)?(?:\*\*)?Step\s*(\d+)[:\.\s]*(?:\*\*)?\s*',
            re.IGNORECASE | re.MULTILINE
        )
        
        matches = list(step_pattern.finditer(text))
        
        if matches:
            for i, match in enumerate(matches):
                step_num = int(match.group(1))
                start = match.end()
                # Get content until next step or end
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                content = text[start:end].strip()
                
                # Clean up the content
                content = self._clean_step_content(content)
                
                # Skip empty steps
                if not content:
                    continue
                
                steps.append(ReasoningStep(
                    step_number=step_num,
                    text=content
                ))
        
        return steps
    
    def _parse_numbered_list(self, text: str) -> List[ReasoningStep]:
        """Parse numbered list format like '1. ', '2. '"""
        steps = []
        
        # Pattern for "1. " or "1) " at start of line
        list_pattern = re.compile(r'(?:^|\n)\s*(\d+)[.\)]\s+(.+?)(?=(?:\n\s*\d+[.\)]|\n\n|\Z))', re.DOTALL)
        
        matches = list(list_pattern.finditer(text))
        
        if matches:
            for match in matches:
                step_num = int(match.group(1))
                content = match.group(2).strip()
                
                # Clean up content
                content = self._clean_step_content(content)
                
                if not content:
                    continue
                
                steps.append(ReasoningStep(
                    step_number=step_num,
                    text=content
                ))
        
        return steps
    
    def _parse_paragraphs(self, text: str) -> List[ReasoningStep]:
        """Parse by splitting into logical paragraphs."""
        steps = []
        
        # Split by double newlines
        paragraphs = re.split(r'\n\s*\n', text)
        
        step_num = 0
        for para in paragraphs:
            para = para.strip()
            
            # Skip empty or very short paragraphs
            if not para or len(para) < 10:
                continue
            
            # Skip common non-step content
            if para.lower().startswith(('let\'s solve', 'solution:', 'therefore', 'final answer')):
                continue
            
            step_num += 1
            content = self._clean_step_content(para)
            
            if not content:
                continue
            
            steps.append(ReasoningStep(
                step_number=step_num,
                text=content
            ))
            
            # Limit to 10 steps max
            if step_num >= 10:
                break
        
        return steps
    
    def _clean_step_content(self, content: str) -> str:
        """Clean individual step content."""
        # Remove markdown markers
        content = re.sub(r'^#{1,3}\s*', '', content)  # Remove ### at start
        content = re.sub(r'\*\*(.+?)\*\*', r'\1', content)  # Remove bold
        content = re.sub(r'^\s*-\s+', '', content, flags=re.MULTILINE)  # Remove bullet points
        
        # Remove LaTeX markers but keep content
        content = re.sub(r'\\\[|\\\]', '', content)
        content = re.sub(r'\\text\{([^}]*)\}', r'\1', content)
        
        # Remove trailing "###" or similar
        content = re.sub(r'\s*#{1,3}\s*$', '', content)
        
        # Clean up whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = content.strip()
        
        return content
    
    def _extract_final_answer(self, text: str) -> Optional[str]:
        """Extract the final numerical answer from text."""
        lines = text.split('\n')
        last_lines = '\n'.join(lines[-30:]) if len(lines) > 30 else text
        search_text = last_lines[-1000:] if len(last_lines) > 1000 else last_lines
        
        # Helper for nested braces
        def extract_boxed_content(s):
            if "\\boxed{" not in s: return None
            start_indices = [m.start() for m in re.finditer(r"\\boxed\{", s)]
            for start_idx in reversed(start_indices):
                idx = start_idx + 7
                depth = 1
                content = ""
                for i in range(idx, len(s)):
                    if s[i] == '{': depth += 1
                    elif s[i] == '}': depth -= 1
                    if depth == 0: return content
                    content += s[i]
            return None

        boxed_content = extract_boxed_content(text)
        if boxed_content:
            cleaned = boxed_content.replace("$", "").replace("\\", "").strip()
            return cleaned

        explicit_match = re.search(r'FINAL_ANSWER:\s*(.*?)(?:\n|$)', text, re.IGNORECASE)
        if explicit_match:
            candidate = explicit_match.group(1).strip()
            if '=' in candidate:
                candidate = candidate.split('=')[-1].strip()
            num_match = re.search(r'(?<![a-zA-Z])(\-?\d+(?:[\.,]\d+)*(?:\/\d+)?)', candidate)
            if num_match:
                return num_match.group(1).replace(',', '')
            let_match = re.search(r'([A-E])', candidate, re.IGNORECASE)
            if let_match:
                return let_match.group(1).upper()

        aqua_match = re.search(r'(?:answer|option)\s+is\s+(?:[\*\s]*)[\(]*([A-E])[\)]*', search_text, re.IGNORECASE)
        if aqua_match:
            start = aqua_match.end()
            following_text = search_text[start:start+20]
            num_match = re.search(r'(?<![a-zA-Z])(\-?\d+(?:[\.,]\d+)*(?:\/\d+)?)', following_text)
            if num_match:
                return num_match.group(1).replace(',', '')
            return aqua_match.group(1).upper()

        final_answer_section = re.search(r'Final\s+Answer[:\s]+(.*?)(?:\.|$|\n)', search_text, re.IGNORECASE | re.DOTALL)
        if final_answer_section:
            answer_text = final_answer_section.group(1)
            if '=' in answer_text:
                answer_text = answer_text.split('=')[-1].strip()
            dollar_match = re.search(r'\$(-?[0-9,]+(?:\.[0-9]+)?)', answer_text)
            if dollar_match:
                return dollar_match.group(1).replace(',', '')
            frac_match = re.search(r'(-?\d+\/\d+)', answer_text)
            if frac_match:
                return frac_match.group(1)
            number_match = re.search(r'(?<![a-zA-Z])(-?[0-9,]+(?:\.[0-9]+)?)', answer_text)
            if number_match:
                return number_match.group(1).replace(',', '')

        bold_answer_patterns = [
            r'\*\*Final\s+Answer[:*\s]+\*?([^\*]+)',
            r'Final\s+Answer[:\s]+\*\*([^\*]+)\*\*',
        ]
        for pattern in bold_answer_patterns:
            match = re.search(pattern, search_text, re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                num_match = re.search(r'(?<![a-zA-Z])(\-?\d+(?:[\.,]\d+)*(?:\/\d+)?)', candidate)
                if num_match:
                    return num_match.group(1).replace(',', '')
                let_match = re.search(r'([A-E])', candidate, re.IGNORECASE)
                if let_match:
                    return let_match.group(1).upper()

        dollar_matches = list(re.finditer(r'\$(-?[0-9,]+(?:\.[0-9]+)?)', search_text))
        if dollar_matches:
            return dollar_matches[-1].group(1).replace(',', '')
        
        patterns = [
            r'final\s+answer[:\s]+\$?(-?[0-9,]+(?:\.[0-9]+)?)',
            r'(?<![a-zA-Z])\*\*(\-?\d+(?:[\.,]\d+)*(?:\/\d+)?)\s*(?:meters?|dollars?|cups?|bolts?)?\*\*',
        ]
        for pattern in patterns:
            match = re.search(pattern, search_text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).replace(',', '')
        
        last_3_lines = '\n'.join(lines[-3:]) if len(lines) >= 3 else text
        number_matches = list(re.finditer(r'(?<![a-zA-Z])(\-?\d+(?:[\.,]\d+)*(?:\/\d+)?)', last_3_lines))
        if number_matches:
            valid_matches = []
            for m in number_matches:
                line_start = last_3_lines.rfind('\n', 0, m.start()) + 1
                if m.start() == line_start and m.end() < len(last_3_lines) and last_3_lines[m.end()] == '.':
                    continue
                valid_matches.append(m)
            if valid_matches:
                return valid_matches[-1].group(1).replace(',', '')
        
        return None


def test_structured_reasoning():
    """Test the structured reasoning generator."""
    gen = StructuredReasoningGenerator(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        debug=True
    )
    
    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    
    trace = gen.generate(question, use_structured_prompt=True)
    
    print("\n" + "="*60)
    print("FINAL RESULT:")
    print("="*60)
    print(f"Question: {trace.question[:100]}...")
    print(f"Final Answer: {trace.final_answer}")
    print(f"Number of Steps: {len(trace.steps)}")
    print(f"Parsing Issues: {trace.parsing_issues}")


if __name__ == "__main__":
    test_structured_reasoning()

