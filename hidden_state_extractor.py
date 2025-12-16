"""
Extract features from the model's internal representations for each reasoning step.

We can extract different types of features:
- Hidden states from different layers
- Attention patterns
- Token probabilities/logits
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Dict, Optional
import numpy as np
try:
    from scipy.stats import entropy as scipy_entropy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    # Fallback entropy calculation
    def scipy_entropy(pk, base=2):
        """Simple entropy calculation if scipy not available."""
        pk = np.asarray(pk)
        pk = pk / pk.sum()
        return -np.sum(pk * np.log2(pk + 1e-10))


class HiddenStateExtractor:
    """Extract hidden states from model for each reasoning step."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-Math-1.5B", device: str = None,
                 use_enhanced_features: bool = True, 
                 layer_indices: Optional[List[int]] = None,
                 feature_config: Optional[Dict] = None):
        """
        Initialize the hidden state extractor.
        
        Args:
            model_name: HuggingFace model name (e.g., "Qwen/Qwen2.5-Math-1.5B", "gpt2")
            device: Device to run on ('cuda' or 'cpu')
            use_enhanced_features: If True, extract multi-layer, attention, entropy, logits
            layer_indices: Which layers to extract (None = auto-select based on model depth)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_enhanced_features = use_enhanced_features
        
        # Which features to extract - can be configured per experiment
        # Default is to use everything
        self.feature_config = feature_config or {
            'multi_layer_last_token': True,  # Just the last token from each layer
            'single_layer_last_token': False,  # Only last token from final layer (fastest)
            'multi_layer_mean_pooled': True,  # Average across all tokens in the step
            'attention': True,
            'logits': True
        }
        
        # Use Auto classes for broader model support
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.model.to(self.device)
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Figure out which layers to extract from
        total_layers = self.model.config.num_hidden_layers
        if layer_indices is None:
            # Pick a few layers spread across the model: early, middle, late, and final
            if total_layers >= 4:
                self.layer_indices = [
                    max(0, total_layers // 4),
                    max(0, total_layers // 2),
                    max(0, 3 * total_layers // 4),
                    total_layers - 1  # Always include the last layer
                ]
            else:
                self.layer_indices = [total_layers - 1]  # Small model, just use last layer
        else:
            self.layer_indices = layer_indices
        
        # Clean up: remove duplicates and sort
        self.layer_indices = sorted(list(set(self.layer_indices)))
        
        # Calculate feature dimension based on configuration
        hidden_size = self.model.config.hidden_size
        if self.use_enhanced_features and self.feature_config:
            dim = 0
            if self.feature_config.get('multi_layer_last_token', False):
                dim += (len(self.layer_indices) + 1) * hidden_size  # All layers + last token
            if self.feature_config.get('single_layer_last_token', False):
                dim += hidden_size  # Just last token from final layer
            if self.feature_config.get('multi_layer_mean_pooled', False):
                dim += len(self.layer_indices) * hidden_size  # Mean pooled layers
            if self.feature_config.get('attention', False):
                dim += 4  # Attention patterns
            if self.feature_config.get('logits', False):
                dim += 3  # Logit confidence features
            self.feature_dim = dim
        elif self.use_enhanced_features:
            # Default: all features
            multi_layer_dim = (len(self.layer_indices) + 1) * hidden_size
            attention_dim = 4
            logit_dim = 3
            self.feature_dim = multi_layer_dim + attention_dim + logit_dim
        else:
            self.feature_dim = hidden_size
    
    def extract_step_hidden_states(self, question: str, reasoning_steps: List[str]) -> List[np.ndarray]:
        """
        Extract hidden states for each reasoning step.
        
        Args:
            question: The original question
            reasoning_steps: List of reasoning step strings
        
        Returns:
            List of feature vectors (one per step). If enhanced features enabled,
            returns concatenated multi-feature vectors based on feature_config.
        """
        if self.use_enhanced_features:
            return self._extract_enhanced_features(question, reasoning_steps)
        else:
            return self._extract_basic_features(question, reasoning_steps)
    
    def _extract_basic_features(self, question: str, reasoning_steps: List[str]) -> List[np.ndarray]:
        """Original basic feature extraction (last layer, last token)."""
        hidden_states_list = []
        
        # Build context incrementally: question + steps up to current step
        accumulated_context = f"Question: {question}\nAnswer: "
        
        for step_idx, step in enumerate(reasoning_steps):
            # Create full context up to this step
            current_text = accumulated_context + step
            
            # Tokenize
            inputs = self.tokenizer(
                current_text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Extract hidden states
            with torch.no_grad():
                outputs = self.model(
                    inputs.input_ids,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # Get hidden states from the last layer
                hidden_states = outputs.hidden_states[-1]
                
                # Use the last token's hidden state
                step_hidden = hidden_states[0, -1, :].cpu().numpy()
                hidden_states_list.append(step_hidden)
            
            # Update accumulated context for next step
            accumulated_context = current_text + " "
        
        return hidden_states_list
    
    def _extract_enhanced_features(self, question: str, reasoning_steps: List[str]) -> List[np.ndarray]:
        """
        Extract enhanced features: multi-layer, attention, entropy, logits, mean pooling.
        
        Returns:
            List of concatenated feature vectors for each step
        """
        feature_vectors = []
        
        # Build context incrementally: question + steps up to current step
        accumulated_context = f"Question: {question}\nAnswer: "
        
        for step_idx, step in enumerate(reasoning_steps):
            # Create full context up to this step
            current_text = accumulated_context + step
            
            # Tokenize
            inputs = self.tokenizer(
                current_text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Tokenize just the step to find its position in the sequence
            step_tokens = self.tokenizer(step, return_tensors="pt", add_special_tokens=False)
            step_token_length = step_tokens.input_ids.shape[1]
            
            # Extract all features
            with torch.no_grad():
                outputs = self.model(
                    inputs.input_ids,
                    output_hidden_states=True,
                    output_attentions=True,
                    return_dict=True
                )
                
                # Get sequence length and step token positions
                seq_len = inputs.input_ids.shape[1]
                # Step tokens are at the end of the sequence
                step_start_idx = seq_len - step_token_length
                step_token_indices = slice(step_start_idx, seq_len)
                
                # 1. HIDDEN STATES
                multi_layer_last_token_features = []
                single_layer_last_token_features = []
                multi_layer_mean_pooled_features = []

                if self.feature_config.get('single_layer_last_token', False):
                    # Only extract last token from final layer (fastest option)
                    single_layer_last_token = outputs.hidden_states[-1][0, -1, :].cpu().numpy()
                    single_layer_last_token_features.append(single_layer_last_token)

                if self.feature_config.get('multi_layer_last_token', True):
                    # Extract last token from each layer
                    for layer_idx in self.layer_indices:
                        layer_hidden = outputs.hidden_states[layer_idx]  # (1, seq_len, hidden_size)
                        last_token = layer_hidden[0, -1, :].cpu().numpy()  # (hidden_size,)
                        multi_layer_last_token_features.append(last_token)
                    # Also include last token from final layer
                    last_token_final = outputs.hidden_states[-1][0, -1, :].cpu().numpy()
                    multi_layer_last_token_features.append(last_token_final)
                
                if self.feature_config.get('multi_layer_mean_pooled', False):
                    # Mean pool across all tokens in step
                    for layer_idx in self.layer_indices:
                        layer_hidden = outputs.hidden_states[layer_idx]  # (1, seq_len, hidden_size)
                        step_hidden = layer_hidden[0, step_token_indices, :]  # (step_len, hidden_size)
                        step_mean = step_hidden.mean(dim=0).cpu().numpy()  # (hidden_size,)
                        multi_layer_mean_pooled_features.append(step_mean)
                
                # 2. ATTENTION WEIGHTS/PATTERNS
                attention_features = []
                if self.feature_config.get('attention', False):
                    if outputs.attentions is not None:
                        # Get attention from last layer (most relevant)
                        last_attention = outputs.attentions[-1]  # (1, num_heads, seq_len, seq_len)
                        last_attention = last_attention[0]  # (num_heads, seq_len, seq_len)
                        
                        # Extract attention for step tokens attending to all tokens
                        step_attention = last_attention[:, step_token_indices, :]  # (num_heads, step_len, seq_len)
                        
                        # Aggregate attention patterns:
                        # a) Average across step tokens first: (num_heads, seq_len)
                        step_attention_avg = step_attention.mean(dim=1)  # (num_heads, seq_len)
                        # Then aggregate across heads and targets
                        attention_features.append(step_attention_avg.mean().cpu().item())
                        attention_features.append(step_attention_avg.std().cpu().item())
                        
                        # b) Attention to question vs answer tokens
                        question_len = seq_len - step_token_length
                        if question_len > 0:
                            question_attention = step_attention[:, :, :question_len].mean().cpu().item()
                            answer_attention = step_attention[:, :, question_len:].mean().cpu().item()
                            attention_features.append(question_attention)
                            attention_features.append(answer_attention)
                        else:
                            # If no question tokens, just use zeros
                            attention_features.extend([0.0, 0.0])
                    else:
                        # Attention not available (e.g., SDPA), use zeros
                        attention_features = [0.0, 0.0, 0.0, 0.0]
                
                # 3. TOKEN PROBABILITY/LOGITS (model confidence) - NO ENTROPY
                logit_features = []
                if self.feature_config.get('logits', False) and hasattr(outputs, 'logits'):
                    step_logits = outputs.logits[0, step_token_indices, :]  # (step_len, vocab_size)
                    # Get probabilities
                    step_probs = torch.softmax(step_logits, dim=-1)  # (step_len, vocab_size)
                    
                    # Extract confidence features (removed entropy)
                    max_probs = step_probs.max(dim=-1)[0].cpu().numpy()  # (step_len,)
                    logit_features.append(max_probs.mean())  # Mean confidence
                    logit_features.append(max_probs.std())   # Confidence variance
                    logit_features.append(max_probs.min())   # Min confidence
            
            # Concatenate features based on configuration
            feature_parts = []

            if self.feature_config.get('single_layer_last_token', False):
                feature_parts.append(single_layer_last_token_features[0]) 

            if self.feature_config.get('multi_layer_last_token', False):
                feature_parts.append(np.concatenate(multi_layer_last_token_features))
            
            if self.feature_config.get('multi_layer_mean_pooled', False):
                if len(multi_layer_mean_pooled_features) > 0:
                    feature_parts.append(np.concatenate(multi_layer_mean_pooled_features))
            
            if self.feature_config.get('attention', False):
                feature_parts.append(np.array(attention_features))
            
            if self.feature_config.get('logits', False):
                feature_parts.append(np.array(logit_features))
            
            # Concatenate all enabled features
            if len(feature_parts) > 0:
                all_features = np.concatenate(feature_parts)
            else:
                # Fallback: use last token from last layer
                all_features = outputs.hidden_states[-1][0, -1, :].cpu().numpy()
            
            feature_vectors.append(all_features)
            
            # Update accumulated context for next step
            accumulated_context = current_text + " "
        
        return feature_vectors
    
    def extract_all_hidden_states(self, question: str, full_reasoning: str) -> Tuple[List[np.ndarray], List[int]]:
        """
        Extract hidden states for all tokens and identify step boundaries.
        
        Args:
            question: The original question
            full_reasoning: Complete reasoning text
        
        Returns:
            Tuple of (list of hidden states per token, list of step boundary indices)
        """
        full_text = f"Question: {question}\nAnswer: {full_reasoning}"
        
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                inputs.input_ids,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Get hidden states from the last layer
            hidden_states = outputs.hidden_states[-1]
            
            # Convert to numpy: (seq_len, hidden_size)
            all_hidden_states = hidden_states[0].cpu().numpy()
            
            step_boundaries = self._identify_step_boundaries(full_reasoning, inputs)
            
            return all_hidden_states, step_boundaries
    
    def _identify_step_boundaries(self, reasoning: str, tokenized_inputs) -> List[int]:
        boundaries = []
        return boundaries


