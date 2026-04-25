"""Pallas splash_attention wiring for Llama 3 8B."""
import torch
from torchax import interop

def register_splash_attention(mesh):
    # This is a stub for the Llama 3 port.
    # In a full implementation, this would register a custom attention function
    # in transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS.
    print("[attention] splash_pallas registration (stub)")
    return "eager"
