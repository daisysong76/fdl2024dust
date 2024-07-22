# Example: "BPD_2022-11-10T12:00:00_2022-11-10T12:30:00_sample" is a datapoint corresponding to
# X = 10 SDO images from 2022-11-10T11:40:00 going backwards in time,
# y_regression = BPDdata (2022-11-10T12:00:00, 2022-11-10T12:30:00),
# y_classification = BPDthreshold ( mean ( y_regression ) )

#  hyperparameter (e.g., called n_windows) that defines how many windows we want to divide the data into between T_start and T_end? Then the data we would have is:
# X - SDO - n_windows number of images (these images will be repeated if we have no new SDO image for the timepoints in the window)
# Y - radiation - n_windows time series patches
# With this data we will train the Stretch model by passing in the first window of SDO and radiation to predict the second radiation window. Then we will pass in the first and second window of SDO and radiation to predict the third radiation window, and so on... Does this make sense? (CCing 

import torch
import torch.nn as nn

class SwinTransformer2DFineTuner:
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
    # Initialize the model (assuming it is defined as SwinTransformer)
    model = SwinTransformer(
        # TODO: change to 1
        num_classes=4,  # Number of classes for classification, no use for regression
        # TODO: change to 2
        n_windows = 8   # Defines how many windows to divide the data into.
        patch_size=(4, 4),
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 6, 2], # Number of layers in each stage
        num_heads=[3, 6, 12, 24],
        window_size=(7, 7), # Window size for self-attention
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2, # Stochastic depth
        norm_layer=nn.LayerNorm,
        patch_norm=False,
        frozen_stages=-1,
        use_checkpoint=False
    )

    # Create a fine-tuner for the model
    fine_tuner = SwinTransformer2DFineTuner(model)

    # Print the model to verify which layers are unfrozen
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")

    # Get the optimizer
    optimizer = fine_tuner.get_optimizer(lr=1e-4)
