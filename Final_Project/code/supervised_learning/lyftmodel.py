'''
## Modified by Aswath Muthuselvam
**University of Maryland, College Park**
**Date: April 29, 2022**

'''

import warnings
from typing import Dict, List

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet50

import sys
# insert at 1, 0 is the script path (or '' in REPL)
# sys.path.insert(1, '/home/aswath/Downloads/l5kit-master/examples/planning/model')

from model.modeling import VisionTransformer
import ml_collections


class RasterizedPlanningModel(nn.Module):
    """Raster-based model for planning.
    """

    def __init__(
            self,
            model_arch: str,
            num_input_channels: int,
            num_targets: int,
            weights_scaling: List[float],
            criterion: nn.Module,
            pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.model_arch = model_arch
        self.num_input_channels = num_input_channels
        self.num_targets = num_targets
        self.register_buffer("weights_scaling", torch.tensor(weights_scaling))
        self.pretrained = pretrained
        self.criterion = criterion

        if pretrained and self.num_input_channels != 3:
            warnings.warn("There is no pre-trained model with num_in_channels != 3, first layer will be reset")

        if model_arch == "transformer":
            def get_toy_config():
                """Returns the ViT-B/16 configuration."""
                config = ml_collections.ConfigDict()
                config.patches = ml_collections.ConfigDict({'size': (16, 16)})
                config.hidden_size = 50
                config.transformer = ml_collections.ConfigDict()
                config.transformer.mlp_dim = 50 #10
                config.transformer.num_heads = 10 #5
                config.transformer.num_layers = 20
                config.transformer.attention_dropout_rate = 0.0
                config.transformer.dropout_rate = 0.1
                config.classifier = 'token'
                config.representation_size = None
                return config

            self.model = VisionTransformer(config = get_toy_config(), img_size=224, num_classes=num_targets, in_channels=num_input_channels, vis=True)
            self.model.fc = nn.Linear(in_features=num_targets, out_features=num_targets)

        elif model_arch == "resnet18":
            self.model = resnet18(pretrained=pretrained)
            self.model.fc = nn.Linear(in_features=512, out_features=num_targets)
        elif model_arch == "resnet50":
            self.model = resnet50(pretrained=pretrained)
            self.model.fc = nn.Linear(in_features=2048, out_features=num_targets)
        else:
            raise NotImplementedError(f"Model arch {model_arch} unknown")

        if self.num_input_channels != 3:
            self.model.conv1 = nn.Conv2d(
                self.num_input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # [batch_size, channels, height, width]
        image_batch = data_batch["image"]
        # [batch_size, num_steps * 2]
        outputs, attn = self.model(image_batch)
        # outputs = outputs[0]
        # print("Model outputs: ",outputs.shape)
        batch_size = len(data_batch["image"])
        # print("Batch size outputs: ",batch_size) #12
        # print("Batch view: ",outputs.view(batch_size, -1, 3).shape)

        if self.training:
            if self.criterion is None:
                raise NotImplementedError("Loss function is undefined.")

            # [batch_size, num_steps * 2]
            targets = (torch.cat((data_batch["target_positions"], data_batch["target_yaws"]), dim=2)).view(
                batch_size, -1
            )
            # [batch_size, num_steps]
            target_weights = (data_batch["target_availabilities"].unsqueeze(-1) * self.weights_scaling).view(
                batch_size, -1
            )
            coolprint = lambda *w: [print(x,'=',eval(x)) for x in w]

            # print("target_weights: ", target_weights.shape)
            # print("targets: ", targets.shape)
            # print("outputs: ", outputs.shape)
            loss = torch.mean(self.criterion(outputs, targets) * target_weights)
            train_dict = {"loss": loss}
            return train_dict
        else:
            predicted = outputs.view(batch_size, -1, 3)
            # [batch_size, num_steps, 2->(XY)]
            pred_positions = predicted[:, :, :2]
            # [batch_size, num_steps, 1->(yaw)]
            pred_yaws = predicted[:, :, 2:3]
            eval_dict = {"positions": pred_positions, "yaws": pred_yaws, "attn": attn}
            return eval_dict
