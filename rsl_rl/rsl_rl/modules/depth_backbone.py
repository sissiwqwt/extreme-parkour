import torch
import torch.nn as nn
import sys
import torchvision


class VisualStudentBackbone(nn.Module):
    def __init__(self, base_backbone, env_cfg) -> None:
        super().__init__()
        activation = nn.ELU()
        self.base_backbone = base_backbone
        if env_cfg == None:
            proprio_dim = 53
        else:
            proprio_dim = env_cfg.env.n_proprio
        self.combination_mlp = nn.Sequential(
            nn.Linear(32 + proprio_dim, 128),
            activation,
            nn.Linear(128, 32),
        )
        self.rnn = nn.GRU(input_size=32, hidden_size=512, batch_first=True)
        self.hidden_states = None

    def forward(self, depth_image, proprioception):
        depth_feature = self.base_backbone(depth_image)
        shared_feature = self.combination_mlp(torch.cat((depth_feature, proprioception), dim=-1))
        shared_feature, self.hidden_states = self.rnn(shared_feature[:, None, :], self.hidden_states)
        return shared_feature.squeeze(1)

    def detach_hidden_states(self):
        self.hidden_states = self.hidden_states.detach().clone()


class DepthLatentHead(nn.Module):
    def __init__(self, latent_dim=32) -> None:
        super().__init__()
        self.output_mlp = nn.Sequential(
            nn.Linear(512, latent_dim),
            nn.Tanh(),
        )

    def forward(self, shared_feature):
        return self.output_mlp(shared_feature)


class HeadingPredictorHead(nn.Module):
    def __init__(self, heading_dim=2) -> None:
        super().__init__()
        self.output_mlp = nn.Sequential(
            nn.Linear(512, heading_dim),
            nn.Tanh(),
        )

    def forward(self, shared_feature):
        return self.output_mlp(shared_feature)

class RecurrentDepthBackbone(nn.Module):
    def __init__(self, base_backbone, env_cfg, heading_dim=2) -> None:
        super().__init__()
        self.heading_dim = heading_dim
        self.visual_backbone = VisualStudentBackbone(base_backbone, env_cfg)
        self.depth_latent_head = DepthLatentHead(latent_dim=32)
        self.heading_predictor_head = HeadingPredictorHead(heading_dim=self.heading_dim)

    def forward(self, depth_image, proprioception):
        shared_feature = self.visual_backbone(depth_image, proprioception)
        depth_latent = self.depth_latent_head(shared_feature)
        heading_pred = self.heading_predictor_head(shared_feature)
        return torch.cat((depth_latent, heading_pred), dim=-1)

    def detach_hidden_states(self):
        self.visual_backbone.detach_hidden_states()

    def heading_parameters(self, include_backbone=True):
        modules = [self.heading_predictor_head]
        if include_backbone:
            modules.insert(0, self.visual_backbone)
        for module in modules:
            yield from module.parameters()

    def action_parameters(self, include_backbone=False):
        modules = [self.depth_latent_head]
        if include_backbone:
            modules.insert(0, self.visual_backbone)
        for module in modules:
            yield from module.parameters()

    def set_heading_trainable(self, trainable):
        for param in self.visual_backbone.parameters():
            param.requires_grad = trainable
        for param in self.heading_predictor_head.parameters():
            param.requires_grad = trainable

    def set_action_trainable(self, trainable, include_backbone=False):
        if include_backbone:
            for param in self.visual_backbone.parameters():
                param.requires_grad = trainable
        for param in self.depth_latent_head.parameters():
            param.requires_grad = trainable

    def load_state_dict(self, state_dict, strict=True):
        state_dict = self._upgrade_legacy_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=strict)

    def _upgrade_legacy_state_dict(self, state_dict):
        if 'output_mlp.0.weight' not in state_dict:
            return state_dict

        output_weight = state_dict['output_mlp.0.weight']
        output_bias = state_dict['output_mlp.0.bias']
        if output_weight.shape[0] != 32 + self.heading_dim:
            return state_dict

        upgraded = {}
        for key, value in state_dict.items():
            if key.startswith('base_backbone.'):
                upgraded['visual_backbone.' + key] = value
            elif key.startswith('combination_mlp.'):
                upgraded['visual_backbone.' + key] = value
            elif key.startswith('rnn.'):
                upgraded['visual_backbone.' + key] = value
            elif key == 'output_mlp.0.weight':
                upgraded['depth_latent_head.output_mlp.0.weight'] = value[:32]
                upgraded['heading_predictor_head.output_mlp.0.weight'] = value[32:]
            elif key == 'output_mlp.0.bias':
                upgraded['depth_latent_head.output_mlp.0.bias'] = value[:32]
                upgraded['heading_predictor_head.output_mlp.0.bias'] = value[32:]
            else:
                upgraded[key] = value
        return upgraded

class StackDepthEncoder(nn.Module):
    def __init__(self, base_backbone, env_cfg) -> None:
        super().__init__()
        activation = nn.ELU()
        self.base_backbone = base_backbone
        self.combination_mlp = nn.Sequential(
                                    nn.Linear(32 + env_cfg.env.n_proprio, 128),
                                    activation,
                                    nn.Linear(128, 32)
                                )

        self.conv1d = nn.Sequential(nn.Conv1d(in_channels=env_cfg.depth.buffer_len, out_channels=16, kernel_size=4, stride=2),  # (30 - 4) / 2 + 1 = 14,
                                    activation,
                                    nn.Conv1d(in_channels=16, out_channels=16, kernel_size=2), # 14-2+1 = 13,
                                    activation)
        self.mlp = nn.Sequential(nn.Linear(16*14, 32), 
                                 activation)
        
    def forward(self, depth_image, proprioception):
        # depth_image shape: [batch_size, num, 58, 87]
        depth_latent = self.base_backbone(None, depth_image.flatten(0, 1), None)  # [batch_size * num, 32]
        depth_latent = depth_latent.reshape(depth_image.shape[0], depth_image.shape[1], -1)  # [batch_size, num, 32]
        depth_latent = self.conv1d(depth_latent)
        depth_latent = self.mlp(depth_latent.flatten(1, 2))
        return depth_latent

    
class DepthOnlyFCBackbone58x87(nn.Module):
    def __init__(self, prop_dim, scandots_output_dim, hidden_state_dim, output_activation=None, num_frames=1):
        super().__init__()

        self.num_frames = num_frames
        activation = nn.ELU()
        self.image_compression = nn.Sequential(
            # [1, 58, 87]
            nn.Conv2d(in_channels=self.num_frames, out_channels=32, kernel_size=5),
            # [32, 54, 83]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [32, 27, 41]
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            activation,
            nn.Flatten(),
            # [32, 25, 39]
            nn.Linear(64 * 25 * 39, 128),
            activation,
            nn.Linear(128, scandots_output_dim)
        )

        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = activation

    def forward(self, images: torch.Tensor):
        images_compressed = self.image_compression(images.unsqueeze(1))
        latent = self.output_activation(images_compressed)

        return latent
