import logging
import torch
from dataclasses import dataclass, field, asdict
from typing import Dict
import math


@dataclass
class InterventionDirectionComponent:
    mean_diff: torch.Tensor = field(default_factory=torch.Tensor)
    normalized_scale: float = field(default_factory=float)
    mean_pos: float = field(default_factory=float)
    mean_neg: float = field(default_factory=float)
    mean_all: float = field(default_factory=float)
    std_pos: float = field(default_factory=float)
    std_neg: float = field(default_factory=float)
    pos_ratio: float = field(default_factory=float)


@dataclass
class InterventionDirection:
    components: Dict[str, InterventionDirectionComponent]
    handles: Dict[str, torch.utils.hooks.RemovableHandle] = field(default_factory=dict)

    def save(self, path):
        torch.save(asdict(self), path)

    @staticmethod
    def load(path):
        ckpt = torch.load(path)
        intervention_dir = InterventionDirection(components={})
        for component, component_results in ckpt['components'].items():
            intervention_dir.components[component] = InterventionDirectionComponent(**component_results)
        return intervention_dir

    def add_intervention(self, model, weight, type="additive", condition_tokens=None, components=None,
                         probe_save_dir=None, step_token_ids=None, debug=False, confidence_threshold=6,
                         normalize_steer_vec=False):
        weight_manager = None
        if type == "probe_last_token" or type == "probe_last_token_mid_reflect" or type.startswith("probe_last_token_temp_"):
            # Initialize weight manager
            monitor = ProbeMonitoringManager(model, probe_save_dir, use_last_token_embedding=True,
                                            intervention_dir=self)
            
            # Parse intervention type to determine scaler configuration
            if type == "probe_last_token":
                scaler = "sigmoid"
            elif type == "probe_last_token_mid_reflect":
                scaler = "mid_reflect"
            elif type.startswith("probe_last_token_temp_"):
                # Format: probe_last_token_temp_<temp>_bias_<bias>
                # Extract temp and bias values and create scaler string
                parts = type.split("_")
                try:
                    temp_idx = parts.index("temp")
                    bias_idx = parts.index("bias")
                    temp = parts[temp_idx + 1] if temp_idx + 1 < len(parts) else "20"
                    bias = parts[bias_idx + 1] if bias_idx + 1 < len(parts) else "6"
                    scaler = f"sigmoid_temp_{temp}_bias_{bias}"
                except (ValueError, IndexError):
                    # Fallback to default sigmoid if parsing fails
                    scaler = "sigmoid"
            else:
                scaler = "sigmoid"
                
            weight_manager = UncertaintyManager(model, monitor, max_intervention=weight, scaler=scaler)
        elif type == "step_confidence" or type.startswith("step_confidence_"):
            # Parse k parameter if provided (format: step_confidence_k_<k_value>)
            k = 5  # default
            if type.startswith("step_confidence_k_"):
                parts = type.split("_")
                try:
                    k_idx = parts.index("k")
                    if k_idx + 1 < len(parts):
                        k = int(parts[k_idx + 1])
                except (ValueError, IndexError):
                    pass  # Use default k=5
            
            # Initialize step-based token confidence weight manager
            weight_manager = LastStepAvgTokenConfidenceWeightManager(
                model, None, max_intervention=weight, k=k, step_token_ids=step_token_ids, debug=debug, confidence_threshold=confidence_threshold
            )
            
            # Create a conditional manager to capture input tokens
            token_manager = ConditionalInterventionManager(model, [])  # Empty list, just for token capture
            self.handles["step_confidence_token_manager"] = model.model.embed_tokens.register_forward_hook(token_manager)
            
            # Pass the token manager to the weight manager
            weight_manager.token_manager = token_manager
            
            # Register the weight manager
            self.handles["step_confidence_manager"] = model.model.register_forward_hook(weight_manager)
        if condition_tokens is not None:
            manager = ConditionalInterventionManager(model, condition_tokens)
            self.handles["manager"] = model.model.embed_tokens.register_forward_hook(manager)
        for component in components if components is not None else self.components:
            if component not in self.components:
                continue
            # Optionally normalize the steering vector to unit norm
            base_direction = self.components[component].mean_diff
            if normalize_steer_vec:
                norm = base_direction.norm()
                base_direction = base_direction / (norm + 1e-12)
            if type == "additive":
                hook = LinearInterventionHook(base_direction, weight)
            elif type == "multiplicative":
                hook = MultiplicativeInterventionHook(base_direction, weight)
            elif type == "activate":
                target = self.components[component].mean_pos / self.components[component].mean_diff.norm()
                hook = TargetedInterventionHook(base_direction, target, weight)
            elif type == "suppress":
                target = self.components[component].mean_neg / self.components[component].mean_diff.norm()
                hook = TargetedInterventionHook(base_direction, target, weight)
            elif type == "probe_last_token" or type == "probe_last_token_mid_reflect" or type.startswith("probe_last_token_temp_"):
                hook = FlexLinearInterventionHook(base_direction, weight_manager)
            elif type == "step_confidence" or type.startswith("step_confidence_"):
                hook = FlexLinearInterventionHook(base_direction, weight_manager)
            if condition_tokens is not None:
                hook = ConditionalInterventionHook(hook, manager)
            self.handles[component] = eval(f"model.{component}.register_forward_hook(hook)")
        return weight_manager

    def add_prober(self, model):
        cacher = ActivationProbe()
        cacher.register_model(model, self)
        return cacher

    def remove_intervention(self):
        for component in self.handles:
            self.handles[component].remove()
        if "manager" in self.handles:
            self.handles["manager"].remove()
        self.handles = {}


class SaveHook():
    def __init__(self, name, act_store):
        self.name = name
        self.act_store = act_store
    
    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        if len(output.shape) == 3:
            output = output.squeeze(0)
        self.act_store[self.name].append(output.cpu())


class ProbeHook():
    def __init__(self, name, direction, act_store):
        self.name = name
        self.direction = direction
        self.act_store = act_store
    
    def __call__(self, module, input, output):
        projection = output @ self.direction.to(output.device)
        self.act_store[self.name].append(projection.cpu())


class TokenEntropyWeightManager():
    def __init__(self, model, monitor, max_intervention=1):
        self.monitor = monitor
        self.max_intervention = max_intervention
        self.intv_strength = 0
    
    def __call__(self, module, input, output):
        # This manager should be hooked to the logits output of the model
        entropy = torch.distributions.Categorical(logits=output).entropy()
        # Only intervene the top_entropy_tokens
        self.intv_strength = torch.where(entropy > math.log(2), self.max_intervention, 0)

def confidence_scaler(conf, threshold):
    # Output intervention strength from confidence score
    return torch.where(conf > threshold, 1, 0)

class TokenConfidenceWeightManager():
    def __init__(self, model, monitor, max_intervention=1, k=5, confidence_threshold=6):
        self.monitor = monitor
        self.max_intervention = max_intervention
        self.intv_strength = 0
        self.k = k
        self.confidence_threshold = confidence_threshold
    
    def __call__(self, module, input, output):
        # This manager should be hooked to the logits output of the model
        confidence = torch.softmax(output, dim=-1)
        # Only intervene the top_confidence_tokens
        topk_confidence, topk_indices = torch.topk(confidence, self.k, dim=-1)
        token_confidence = -1 / self.k * torch.log(topk_confidence).sum(dim=-1)
        self.intv_strength = confidence_scaler(token_confidence, self.confidence_threshold) * self.max_intervention

class AvgTokenConfidenceWeightManager():
    def __init__(self, model, monitor, max_intervention=1, k=5, debug=False):
        self.model = model
        self.monitor = monitor
        self.max_intervention = max_intervention
        self.current_length = 0
        self.intv_strength = 0
        self.avg_confidence = 0
        self.k = k
        self.debug = debug
        if self.debug:
            self.history = []
    def __call__(self, module, input, output):
        # This manager should be hooked to the logits output of the model
        hidden_states = output[0]
        logits = self.model.lm_head(hidden_states)
        confidence = torch.softmax(logits, dim=-1)
        # Only intervene the top_confidence_tokens
        topk_confidence, topk_indices = torch.topk(confidence, self.k, dim=-1)
        token_confidence = -1 / self.k * torch.log(topk_confidence).sum(dim=-1)
        self.avg_confidence += token_confidence
        self.current_length += 1
        self.intv_strength = confidence_scaler(self.avg_confidence / self.current_length) * self.max_intervention
        if self.debug:
            self.history.append(self.intv_strength)

    def clear(self):
        self.current_length = 0
        self.avg_confidence = 0
        self.intv_strength = 0
        if self.debug:
            history = self.history
            self.history = []
            return history
    
class LastStepAvgTokenConfidenceWeightManager(AvgTokenConfidenceWeightManager):
    def __init__(self, model, monitor, max_intervention=1, k=5, step_token_ids=None, debug=False, confidence_threshold=6):
        super().__init__(model, monitor, max_intervention, k, debug)
        self.step_token_ids = step_token_ids
        self.confidence_threshold = confidence_threshold
        # Initialize per-sequence tracking
        self.current_length_per_seq = None
        self.avg_confidence_per_seq = None
        self.model = model
        self.token_manager = None  # Will be set by add_intervention

    def __call__(self, module, input, output):
        # This manager is hooked to the model.model output
        hidden_states = output[0]
        logits = self.model.lm_head(hidden_states)
        
        # We are interested in the confidence of the last token for the current step
        last_token_logits = logits[:, -1, :]
        
        confidence = torch.softmax(last_token_logits, dim=-1)
        
        # Only intervene the top_confidence_tokens
        topk_confidence, topk_indices = torch.topk(confidence, self.k, dim=-1)
        token_confidence_score = -1 / self.k * torch.log(topk_confidence + 1e-6).sum(dim=-1)
        
        # Initialize per-sequence tracking if needed
        if self.current_length_per_seq is None:
            batch_size = token_confidence_score.shape[0] if token_confidence_score.ndim > 0 else 1
            print(f"Initializing per-sequence tracking for {batch_size} sequences")
            self.current_length_per_seq = torch.zeros(batch_size, device=token_confidence_score.device)
            self.avg_confidence_per_seq = torch.zeros(batch_size, device=token_confidence_score.device)
        
        # Update per-sequence averages
        self.avg_confidence_per_seq = (self.avg_confidence_per_seq * self.current_length_per_seq + token_confidence_score) / (self.current_length_per_seq + 1)
        self.current_length_per_seq += 1
        
        self.intv_strength = confidence_scaler(self.avg_confidence_per_seq, self.confidence_threshold) * self.max_intervention
        if self.debug:
            self.history.append(torch.cat([self.intv_strength]))
        # Check for step tokens and reset specific sequences
        if (self.step_token_ids is not None and 
            self.token_manager is not None and 
            self.token_manager.current_input_tokens is not None):
            
            # Get the last token from each sequence in the batch
            current_tokens = self.token_manager.current_input_tokens
            last_tokens = current_tokens[:, -1]
            
            # Convert step_token_ids to tensor if it's not already
            if not isinstance(self.step_token_ids, torch.Tensor):
                step_tokens_tensor = torch.tensor(self.step_token_ids, device=last_tokens.device)
            else:
                step_tokens_tensor = self.step_token_ids.to(last_tokens.device)
            
            # Check which sequences' last token is a step token
            reset_mask = torch.isin(last_tokens, step_tokens_tensor)
            
            # Reset only the sequences with step tokens
            self.current_length_per_seq[reset_mask] = 0
            self.avg_confidence_per_seq[reset_mask] = 0
    
    def clear(self):
        self.current_length_per_seq = None
        self.avg_confidence_per_seq = None
        return super().clear()

# This hook is attached to the o_proj module
class Qwen2CaptureAttnContributionHook():
    def __init__(self, name, act_store, num_heads, head_dim, hidden_size):
        self.name = name
        self.act_store = act_store
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size
    
    def __call__(self, module, input, output):
        attn_out = input[0].detach()
        attn_out = attn_out.reshape(attn_out.size(0), self.num_heads, self.head_dim)
        o_proj = module.weight.detach().clone()
        o_proj = o_proj.reshape(self.hidden_size, self.num_heads, self.head_dim).permute(1, 2, 0).contiguous()
        self.act_store[self.name].append(torch.einsum("snk,nkh->snh", attn_out, o_proj).cpu())


class LinearInterventionHook():
    def __init__(self, direction, weight):
        self.direction = direction
        self.weight = weight
    
    def __call__(self, module, input, output):
        self.direction = self.direction.type_as(output[0] if isinstance(output, tuple) else output)
        if isinstance(output, tuple):
            output = (output[0] + self.direction.to(output[0].device) * self.weight, output[1])
        else:
            output = output + self.direction.to(output.device) * self.weight
        return output


class FlexLinearInterventionHook():
    def __init__(self, direction, manager):
        self.direction = direction
        self.manager = manager
    
    def __call__(self, module, input, output):
        remains = None
        self.direction = self.direction.type_as(output[0] if isinstance(output, tuple) else output)
        weights = self.manager.intv_strength
        if isinstance(output, tuple):
            remains = output[1:]
            output = output[0]
        if isinstance(weights, torch.Tensor):
            if weights.ndim == 0:
                weights = weights.unsqueeze(0).type_as(output)
            else:
                weights = weights[:, None].type_as(output)
            weights = weights.to(output.device)
        try:
            output = output + (self.direction.to(output.device)[None, :] * weights)[:, None, :]
        except Exception as e:
            raise e
        if remains is not None:
            output = (output, remains)
        return output


class MultiplicativeInterventionHook():
    def __init__(self, direction, weight):
        self.direction = direction / direction.norm()
        self.weight = weight
    
    def __call__(self, module, input, output):
        projection = output @ self.direction.to(output.device)
        output = output + self.weight * torch.outer(projection, self.direction.to(output.device))
        return output


class ConditionalInterventionManager():
    def __init__(self, model, activate_tokens, debug=False):
        self.is_active = None
        self.current_input_tokens = None
        self.activate_tokens = torch.tensor(activate_tokens).cuda() if activate_tokens else None
        self.debug = debug
        if self.debug:
            self.history = []

    def __call__(self, module, input, output):
        self.current_input_tokens = input[0]  # Store current input tokens
        if self.activate_tokens is not None:
            self.is_active = torch.isin(input[0], self.activate_tokens).to(input[0].device)
        if self.debug:
            self.history.append(self.is_active)


class UQWeightSigmoidScaler():
    def __init__(self, max_intervention=1, temp=20, bias=6):
        self.max_intervention = max_intervention
        self.temp = temp
        self.bias = bias

    def __call__(self, score):
        return self.max_intervention * (-(torch.sigmoid((score - self.bias) / self.temp) - 0.5) * 2)

class UQWeightMidReflectScaler():
    def __init__(self, max_intervention=1):
        self.max_intervention = max_intervention

    def __call__(self, score):
        MID_UQ = 2.5
        return self.max_intervention * ((1 -(torch.sigmoid(torch.abs(score - MID_UQ)) - 0.5) * 4))

class UncertaintyManager():
    def __init__(self, model, monitor, max_intervention=1, scaler="sigmoid"):
        self.monitor = monitor
        self.max_intervention = max_intervention
        self.intv_strength = 0
        
        # Parse scaler type to extract temp and bias for sigmoid scaler
        if scaler == "sigmoid" or scaler.startswith("sigmoid_"):
            temp = 20  # default
            bias = 6   # default
            
            # Parse temp and bias from scaler string if provided
            if scaler.startswith("sigmoid_"):
                # Format: sigmoid_temp_<temp>_bias_<bias>
                parts = scaler.split("_")
                if len(parts) >= 4:
                    try:
                        temp_idx = parts.index("temp")
                        bias_idx = parts.index("bias")
                        if temp_idx + 1 < len(parts):
                            temp = float(parts[temp_idx + 1])
                        if bias_idx + 1 < len(parts):
                            bias = float(parts[bias_idx + 1])
                    except (ValueError, IndexError):
                        # Use defaults if parsing fails
                        pass
            
            self.scaler = UQWeightSigmoidScaler(max_intervention, temp, bias)
        elif scaler == "mid_reflect":
            self.scaler = UQWeightMidReflectScaler(max_intervention)
        else:
            raise ValueError(f"Unsupported scaler: {scaler}")
        model.model.register_forward_pre_hook(self)

    def __call__(self, module, input):
        _, score = self.monitor.get_prediction()
        if score is None:
            return
        self.intv_strength = self.scaler(score)


    def clear(self):
        self.intv_strength = 0
        self.monitor.clear_cache()
        logging.info("UncertaintyManager: cleared")

class ConditionalInterventionHook():
    def __init__(self, base_hook, manager):
        self.base_hook = base_hook
        self.manager = manager
    
    def __call__(self, module, input, output):
        intervened_output = self.base_hook(module, input, output)
        if isinstance(intervened_output, tuple):
            result = (torch.where(self.manager.is_active[..., None], intervened_output[0], output[0]), intervened_output[1])
        else:
            result = torch.where(self.manager.is_active[..., None], intervened_output, output)
        return result


class TargetedInterventionHook():
    def __init__(self, direction, target, weight):
        self.direction = direction / direction.norm()
        self.target = target
        self.weight = weight
    
    def __call__(self, module, input, output):
        projection = output @ self.direction.to(output.device)
        output = output - self.weight * (torch.outer(projection - self.target, self.direction.to(output.device)))
        return output


# This hook is attached to the self_attn.attn module
class Qwen2HeadDisableHook():
    def __init__(self, num_heads, head_dim, hidden_size, disabled_heads):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.disabled_heads = disabled_heads
    
    def __call__(self, module, input, output):
        if len(self.disabled_heads) == 0:
            return output
            
        # Reshape output to [batch, num_heads, head_dim]
        output_reshaped = output.view(-1, self.num_heads, self.head_dim)
        
        # Create mask for disabled heads
        mask = torch.ones(self.num_heads, device=output.device).type_as(output_reshaped)
        mask[self.disabled_heads] = 0
        
        # Apply mask and reshape back
        output_masked = output_reshaped * mask[None, :, None] 
        return output_masked.reshape(-1, self.num_heads * self.head_dim)


# This hook is attached to the self_attn.o_proj module
class Qwen2HeadModifiyHook():
    def __init__(self, num_heads, head_dim, hidden_size, head_indices, direction):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.head_indices = head_indices
        self.direction = direction / direction.norm()
    
    def __call__(self, module, input, output):
        self.direction = self.direction.to(input[0].device)
        # Reshape output to [batch, num_heads, head_dim]
        attn_out = input[0].detach()
        attn_out = attn_out.reshape(attn_out.size(0), self.num_heads, self.head_dim)
        o_proj = module.weight.detach().clone()
        o_proj = o_proj.reshape(self.hidden_size, self.num_heads, self.head_dim).permute(1, 2, 0).contiguous()
        headwise_out = torch.einsum("snk,nkh->snh", attn_out, o_proj)
        target_heads_out = headwise_out[:, self.head_indices, :]
        headwise_out[:, self.head_indices, :] = target_heads_out - self.direction[None, None, :] * (target_heads_out @ self.direction)[:, :, None]
        return headwise_out.sum(dim=1), output[1]


class HeadInterventionManager():
    def __init__(self, target_heads, mode="disable", direction=None):
        """
        target_heads: list of (layer_idx, List[head_idx]), the heads to disable
        """
        self.target_heads = target_heads
        self.mode = mode
        self.handles = {}
        self.direction = direction
    def add_intervention(self, model,):
        for layer_idx, head_idx_list in self.target_heads:
            if self.mode == "disable":
                hook = Qwen2HeadDisableHook(model.model.layers[layer_idx].self_attn.num_heads, 
                                            model.model.layers[layer_idx].self_attn.head_dim, 
                                            model.model.layers[layer_idx].self_attn.hidden_size, 
                                            head_idx_list)
                handle = model.model.layers[layer_idx].self_attn.attn.register_forward_hook(hook)
            elif self.mode == "modify":
                assert self.direction is not None, "Direction is required for modify mode"
                layer_direction = self.direction.components[f"model.layers[{layer_idx}].self_attn"].mean_diff
                hook = Qwen2HeadModifiyHook(model.model.layers[layer_idx].self_attn.num_heads, 
                                            model.model.layers[layer_idx].self_attn.head_dim, 
                                            model.model.layers[layer_idx].self_attn.hidden_size, 
                                            head_idx_list, 
                                            layer_direction)
                handle = model.model.layers[layer_idx].self_attn.o_proj.register_forward_hook(hook)
            self.handles[layer_idx] = handle
    
    def remove_intervention(self):
        for layer_idx in self.handles:
            self.handles[layer_idx].remove()
        self.handles = {}


class ActivationCacher():
    def __init__(self):
        self.cache = {}
    
    def register_model(self, model, target_modules):
        for target_module in target_modules:
            self.cache[target_module] = []
            hook = SaveHook(target_module, self.cache)
            eval(f"model.{target_module}.register_forward_hook(hook)")

    def get_cache(self):
        return self.cache
    
    def clear_cache(self):
        for key in self.cache:
            self.cache[key].clear()


class Qwen2AttentionActivationCacher(ActivationCacher):
    def __init__(self):
        super().__init__()

    def register_model(self, model, target_modules):
        for target_module in target_modules:
            module = eval(f"model.{target_module}")
            self.cache[target_module] = []
            hook = Qwen2CaptureAttnContributionHook(target_module, self.cache, module.num_heads, module.head_dim, module.hidden_size)
            module.o_proj.register_forward_hook(hook)


class ActivationProbe(ActivationCacher):
    def __init__(self):
        super().__init__()

    def register_model(self, model, direction):
        for component in direction.components:
            self.cache[component] = []
            hook = ProbeHook(component, direction.components[component].mean_diff / direction.components[component].mean_diff.norm(), self.cache)
            eval(f"model.{component}.register_forward_hook(hook)")

    def compile_cache(self):
        """Compile the cache into a single tensor and clear it"""
        outputs = []
        for component in self.cache:
            if not self.cache[component]: return None
            outputs.append(torch.cat(self.cache[component], dim=0)) # [num_tokens, batch_size]
        outputs = torch.cat(outputs, dim=2).squeeze(0) # [batch_size, num_features]
        self.clear_cache()
        return outputs


class LastTokenEmbeddingCacher(ActivationCacher):
    def __init__(self):
        super().__init__()


    def register_model(self, model):
        self.cache["last_token_embedding"] = []
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            if len(output.shape) == 2:
                output = output.unsqueeze(0) # Fill in the batch dimension
            self.cache["last_token_embedding"].append(output[:, -1, :].cpu()) # [batch_size, hidden_size]
        model.model.layers[-1].register_forward_hook(hook)

    def compile_cache(self):
        """Compile the cache into a single tensor and clear it"""
        if not self.cache["last_token_embedding"]: return None
        outputs = torch.cat(self.cache["last_token_embedding"], dim=0).squeeze(0) # [batch_size, hidden_size]
        self.clear_cache()
        return outputs


MODEL_NUM_LAYERS_MAP = {
    "deepseek-r1-qwen-1.5b": 28,
    "QwQ-32b-Q8_0": 64,
    "QwQ-32b": 64,
    "deepseek-r1-llama-8b": 32,
    "deepseek-r1-qwen-14b": 48,
    "deepseek-r1-qwen3-8b": 36,
    "thinkedit-llama-8b": 32,
    "thinkedit-qwen-14b": 48,
}
MODEL_LAYER_MAP = {
    key: ["model.layers[{layer_idx}].self_attn".format(layer_idx=i) for i in range(MODEL_NUM_LAYERS_MAP[key])] + \
         ["model.layers[{layer_idx}].mlp".format(layer_idx=i) for i in range(MODEL_NUM_LAYERS_MAP[key])] for key in MODEL_NUM_LAYERS_MAP
}
MODEL_ATTN_LAYER_MAP = {
    key: ["model.layers[{layer_idx}].self_attn".format(layer_idx=i) for i in range(MODEL_NUM_LAYERS_MAP[key])] for key in MODEL_NUM_LAYERS_MAP
}



class ProbeMonitor():
    """Monitor model outputs using saved probe classifiers."""
    
    def __init__(self, probe_save_dir):
        """Load saved classifier weights and bias."""
        self.weights = torch.from_numpy(torch.load(f"{probe_save_dir}/clf_weights.pt"))
        self.bias = torch.from_numpy(torch.load(f"{probe_save_dir}/clf_bias.pt"))
        self.predictions = []
        self.scores = []
    
    def predict(self, features):
        """Apply classifier to features.
        Args:
            features: Tensor of shape (batch_size, num_features)
        Returns:
            prediction: Tensor of shape (batch_size,), 1 if positive, 0 if negative
            score: Tensor of shape (batch_size,), score of the prediction
        """
        features = features.type_as(self.weights).to(self.weights.device)
        score = features @ self.weights.flatten() + self.bias.item()
        prediction = torch.where(score > 0, torch.ones_like(score), torch.zeros_like(score))
        self.predictions.append(prediction)
        self.scores.append(score)
        return prediction, score
    
    def clear(self):
        """Clear prediction history."""
        self.predictions.clear()
        self.scores.clear()


class ProbeMonitoringManager():
    """Manager for monitoring model outputs with saved probe classifiers."""
    
    def __init__(self, model, probe_save_dir, use_last_token_embedding=False, 
                 intervention_dir=None):
        """
        Initialize monitoring manager using existing hooks.
        
        Args:
            model: The model to monitor
            probe_save_dir: Directory containing saved classifier
            use_last_token_embedding: Whether to use last token embeddings
            intervention_dir: InterventionDirection object (for probe directions)
        """
        self.model = model
        self.monitor = ProbeMonitor(probe_save_dir)
        
        if use_last_token_embedding:
            # Use existing LastTokenEmbeddingCacher
            self.cacher = LastTokenEmbeddingCacher()
            self.cacher.register_model(model)
        else:
            # Use existing ActivationProbe  
            if intervention_dir is None:
                raise ValueError("intervention_dir required for probe direction monitoring")
            self.cacher = intervention_dir.add_prober(model)
    
    def get_prediction(self):
        """Get prediction from cached features."""
        features = self.cacher.compile_cache()
        if features is None:
            return None, None
        return self.monitor.predict(features)
    
    def clear_cache(self):
        """Clear caches."""
        self.cacher.clear_cache()
        self.monitor.clear()
    
    def remove_hooks(self):
        """Remove hooks."""
        if hasattr(self.cacher, 'remove_hook'):
            self.cacher.remove_hook()
