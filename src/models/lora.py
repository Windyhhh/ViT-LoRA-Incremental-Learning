import torch
import torch.nn as nn
import numpy as np
from transformers import ViTModel

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=16, alpha=32):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
    
    def forward(self, x):
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


class LoRAViT(nn.Module):
    def __init__(self, model_name, rank=16, alpha=32):
        super().__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        self.hidden_dim = self.vit.config.hidden_size
        
        for param in self.vit.parameters():
            param.requires_grad = False
        
        self.original_forwards = {}
        for i, layer in enumerate(self.vit.encoder.layer):
            attention = layer.attention.attention
            self.original_forwards[f'layer_{i}_q'] = attention.query.forward
            self.original_forwards[f'layer_{i}_v'] = attention.value.forward
        
        self.lora_layers = nn.ModuleDict()
        for i, layer in enumerate(self.vit.encoder.layer):
            self.lora_layers[f'layer_{i}_q'] = LoRALayer(
                self.hidden_dim, self.hidden_dim, rank, alpha
            )
            self.lora_layers[f'layer_{i}_v'] = LoRALayer(
                self.hidden_dim, self.hidden_dim, rank, alpha
            )
        
        self._inject_lora()
    
    def _inject_lora(self):
        for i, layer in enumerate(self.vit.encoder.layer):
            attention = layer.attention.attention
            lora_q = self.lora_layers[f'layer_{i}_q']
            lora_v = self.lora_layers[f'layer_{i}_v']
            original_q_forward = self.original_forwards[f'layer_{i}_q']
            original_v_forward = self.original_forwards[f'layer_{i}_v']
            
            def make_lora_forward(orig_forward, lora_layer):
                def forward(x):
                    return orig_forward(x) + lora_layer(x)
                return forward
            
            attention.query.forward = make_lora_forward(original_q_forward, lora_q)
            attention.value.forward = make_lora_forward(original_v_forward, lora_v)
    
    def forward(self, pixel_values, return_intermediate=False):
        """
        Forward pass with optional intermediate layer features.

        Args:
            pixel_values: Input images
            return_intermediate: If True, return features from intermediate layers

        Returns:
            If return_intermediate=False: final features [batch, 768]
            If return_intermediate=True: dict with features from multiple layers
        """
        outputs = self.vit(pixel_values=pixel_values, output_hidden_states=return_intermediate)
        final_features = outputs.last_hidden_state[:, 0, :]

        if not return_intermediate:
            return final_features

        # 返回多层特征用于多层次蒸馏
        intermediate_features = {}
        hidden_states = outputs.hidden_states  # 包括embedding层和所有encoder层

        # 提取指定层的特征 (DISTILL_LAYERS中的层)
        from configs.config import Config
        for layer_idx in Config.DISTILL_LAYERS:
            if layer_idx < len(hidden_states):
                # 取[CLS]标记的特征
                intermediate_features[f'layer_{layer_idx}'] = hidden_states[layer_idx][:, 0, :]

        return {
            'final': final_features,
            'intermediate': intermediate_features
        }
    
    def get_lora_state_dict(self):
        return {k: v.cpu().clone() for k, v in self.lora_layers.state_dict().items()}
    
    def set_lora_state_dict(self, state_dict):
        model_device = next(self.vit.parameters()).device
        
        needs_reconstruction = False
        for key, param in state_dict.items():
            if key in self.lora_layers.state_dict():
                current_shape = self.lora_layers.state_dict()[key].shape
                if param.shape != current_shape:
                    needs_reconstruction = True
                    break
        
        if needs_reconstruction:
            layer_ranks = {}
            for key in state_dict.keys():
                if key.endswith('.lora_A'):
                    layer_name = key.rsplit('.', 1)[0]
                    rank = state_dict[key].shape[0]
                    layer_ranks[layer_name] = rank
            
            new_lora_layers = nn.ModuleDict()
            for layer_name, rank in layer_ranks.items():
                new_lora_layers[layer_name] = LoRALayer(
                    self.hidden_dim, self.hidden_dim, rank, 
                    self.lora_layers[layer_name].alpha
                ).to(model_device)
            
            self.lora_layers = new_lora_layers
            self._inject_lora()
        
        state_dict_on_device = {k: v.to(model_device) for k, v in state_dict.items()}
        self.lora_layers.load_state_dict(state_dict_on_device)
