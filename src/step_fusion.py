from transformers import get_scheduler
from torch import optim
from lora import LoRALayer  # Import your LoRA implementation


class FusionModel(nn.Module):
    def __init__(self, ...):
        super(FusionModel, self).__init__()
        self.video_model = VideoModel(...)
        self.timeseries_model = TimeSeriesModel(...)
        self.lora_layer = LoRALayer(in_features=1536, out_features=1536)  # Adjust features as per your combined model

    def forward(self, video_x, timeseries_x):
        video_features = self.video_model(video_x)
        timeseries_features = self.timeseries_model(timeseries_x)
        combined_features = torch.cat((video_features, timeseries_features), dim=1)
        combined_features = self.lora_layer(combined_features)  # Apply LoRA Layer
        return combined_features

optimizer = AdamW(model.parameters(), lr=args.lr)
lr_scheduler = get_scheduler(
    name=args.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=args.num_warmup_steps,
    num_training_steps=args.num_training_steps,
)
