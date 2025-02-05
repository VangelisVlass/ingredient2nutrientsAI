import torch
import torch.nn as nn
from transformers import DistilBertModel
from config import CONFIG 

# Define activation functions with explanations
activation_map = {
    "gelu": nn.GELU(),  # Gaussian Error Linear Unit (GELU)
    # ✅ Smooth activation, commonly used in Transformers
    # ✅ Helps with better gradient flow
    # ✅ Default in models like BERT, GPT, and Transformer-based architectures

    "swish": nn.SiLU(),  # Swish (SiLU - Sigmoid Linear Unit)
    # ✅ More flexible than ReLU, adaptive activation
    # ✅ Similar to GELU, smooth and non-monotonic
    # ✅ Helps prevent "dying neurons" (where activations go to zero)

    "relu": nn.ReLU(),  # Rectified Linear Unit (ReLU)
    # ✅ Fast and simple, widely used in deep learning
    # ❌ Can cause "dying ReLU" issue (neurons output zero)
    # ❌ Not as smooth as GELU or Swish

    "tanh": nn.Tanh(),  # Hyperbolic Tangent (Tanh)
    # ✅ Smooth and zero-centered
    # ✅ Useful for outputs in range [-1, 1]
    # ❌ Can saturate (gradient vanishes in extreme values)

    "sigmoid": nn.Sigmoid()  # Sigmoid Activation
    # ✅ S-shaped function, outputs in range [0, 1]
    # ✅ Good for probability-based outputs
    # ❌ Can cause vanishing gradients when values approach 0 or 1
}


# 🏗️ **MODEL DEFINITION**
class ModifiedDistilBERT(nn.Module):
    def __init__(self, output_size=len(CONFIG["nutrients_predicted"])):
        super(ModifiedDistilBERT, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained(CONFIG["model_used"])

        # Load hidden layer configuration from CONFIG
        hidden_layers = CONFIG.get("hidden_layers", [])  
        activation_fn = CONFIG.get("activation_function", "gelu").lower() 
        activation_layer = activation_map.get(activation_fn, nn.GELU()) 

        # Create dynamic feed-forward network
        layers = []
        input_dim = 768 

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation_layer)  
            layers.append(nn.Dropout(CONFIG.get("dropout_rate", 0.1))) 
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, output_size))  
        self.feedforward = nn.Sequential(*layers)

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # Extract [CLS] token representation
        return self.feedforward(cls_output)
