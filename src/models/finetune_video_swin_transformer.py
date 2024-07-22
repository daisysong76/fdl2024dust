import torch
import torch.nn as nn

class SwinTransformer3DFineTuner:
    def __init__(self, model, fine_tune_first_layer=True, fine_tune_last_layer=True):
        self.model = model
        self.fine_tune_first_layer = fine_tune_first_layer
        self.fine_tune_last_layer = fine_tune_last_layer

        self.freeze_all_layers()
        self.unfreeze_selected_layers()

    def freeze_all_layers(self):
        """Freeze all layers in the model."""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_selected_layers(self):
        """Unfreeze the first and last layers."""
        if self.fine_tune_first_layer:
            for param in self.model.patch_embed.parameters():
                param.requires_grad = True

        if self.fine_tune_last_layer:
            for param in self.model.head.parameters():
                param.requires_grad = True

    def get_parameters(self):
        """Get model parameters that require gradients."""
        return filter(lambda p: p.requires_grad, self.model.parameters())

    def get_optimizer(self, lr=1e-4):
        """Get an optimizer for the fine-tuned layers."""
        return torch.optim.Adam(self.get_parameters(), lr=lr)

# Usage example
if __name__ == "__main__":
    # Initialize the model (assuming it is defined as SwinTransformer3D)
    model = SwinTransformer3D(
        num_classes=4,
        patch_size=(4, 4, 4),
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=(2, 7, 7),
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        patch_norm=False,
        frozen_stages=-1,
        use_checkpoint=False
    )

    # Create a fine-tuner for the model
    fine_tuner = SwinTransformer3DFineTuner(model)

    # Print the model to verify which layers are unfrozen
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")

    # Get the optimizer
    optimizer = fine_tuner.get_optimizer(lr=1e-4)
