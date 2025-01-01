from dataclasses import dataclass
from typing import Optional

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import DDPMScheduler
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from peft import LoraConfig
from torch import nn, Tensor
from transformers import AutoTokenizer, CLIPTextModel


@dataclass
class Pix2PixConfig:
    """Configuration for Pix2Pix model initialization."""
    lora_rank_unet: int = 8
    lora_rank_vae: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_id: str = "stabilityai/sd-turbo"
    timesteps: int = 999


class TwinConv(nn.Module):
    """Twin convolution module that interpolates between pretrained and current convolutions."""

    def __init__(self, conv_pretrained: nn.Module, conv_current: nn.Module):
        super().__init__()
        self.conv_pretrained = conv_pretrained.requires_grad_(False)
        self.conv_current = conv_current
        self.interpolation_factor: Optional[float] = None

    def forward(self, x: Tensor) -> Tensor:
        if self.interpolation_factor is None:
            return self.conv_current(x)

        x_pretrained = self.conv_pretrained(x).detach()
        x_current = self.conv_current(x)
        return x_pretrained * (
                1 - self.interpolation_factor) + x_current * self.interpolation_factor


class Pix2PixTurbo(nn.Module):
    """
    Pix2Pix-Turbo model for image-to-image translation.

    This implementation focuses on the base model without pretrained configurations,
    using LoRA for efficient fine-tuning.
    """

    def __init__(self, config: Optional[Pix2PixConfig] = None):
        super().__init__()
        self.config = config or Pix2PixConfig()
        self.device = torch.device(self.config.device)

        # Initialize components
        self._init_scheduler()
        self._init_tokenizer_and_text_encoder()
        self._init_vae()
        self._init_unet()
        self._init_lora_adapters()

        # Setup timesteps
        self.timesteps = torch.tensor([self.config.timesteps], device=self.device).long()

    def _init_scheduler(self):
        """Create a 1-step scheduler for denoising."""
        self.scheduler = DDPMScheduler.from_pretrained(
            "stabilityai/sd-turbo",
            subfolder="scheduler"
        )
        self.scheduler.set_timesteps(1, device=self.device)
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)

    def _init_tokenizer_and_text_encoder(self):
        """Initialize tokenizer and text encoder components."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.config.model_id,
            subfolder="text_encoder"
        ).to(self.device).requires_grad_(False)

    def _init_vae(self):
        """Initialize and configure VAE with skip connections."""
        self.vae = AutoencoderKL.from_pretrained(
            self.config.model_id,
            subfolder="vae"
        ).to(self.device)

        # Add skip connection convolutions
        for i, channels in enumerate([512, 256, 128, 128]):
            skip_conv = nn.Conv2d(
                channels,
                512 if i < 3 else 256,
                kernel_size=1,
                bias=False
            ).to(self.device)
            # Initialize with small weights
            nn.init.constant_(skip_conv.weight, 1e-5)
            setattr(self.vae.decoder, f"skip_conv_{i + 1}", skip_conv)

        # Configure VAE methods
        self.vae.encoder.forward = self._vae_encoder_forward.__get__(
            self.vae.encoder,
            self.vae.encoder.__class__
        )
        self.vae.decoder.forward = self._vae_decoder_forward.__get__(
            self.vae.decoder,
            self.vae.decoder.__class__
        )
        self.vae.decoder.ignore_skip = False
        self.vae.decoder.gamma = 1.0

    def _init_unet(self):
        """Initialize U-Net model."""
        self.unet = UNet2DConditionModel.from_pretrained(
            self.config.model_id,
            subfolder="unet"
        ).to(self.device)

    def _init_lora_adapters(self):
        """Initialize LoRA adapters for VAE and U-Net."""
        # VAE LoRA configuration
        vae_target_modules = [
            "conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
            "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
            "to_k", "to_q", "to_v", "to_out.0",
        ]
        vae_lora_config = LoraConfig(
            r=self.config.lora_rank_vae,
            init_lora_weights="gaussian",
            target_modules=vae_target_modules
        )
        self.vae.add_adapter(vae_lora_config, adapter_name="vae_skip")

        # U-Net LoRA configuration
        unet_target_modules = [
            "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2",
            "conv_shortcut", "conv_out", "proj_in", "proj_out",
            "ff.net.2", "ff.net.0.proj"
        ]
        unet_lora_config = LoraConfig(
            r=self.config.lora_rank_unet,
            init_lora_weights="gaussian",
            target_modules=unet_target_modules
        )
        self.unet.add_adapter(unet_lora_config)

        # Store configurations for later use
        self.target_modules = {
            "vae": vae_target_modules,
            "unet": unet_target_modules
        }
        self.lora_ranks = {
            "vae": self.config.lora_rank_vae,
            "unet": self.config.lora_rank_unet
        }

    @staticmethod
    def _vae_encoder_forward(self, sample: Tensor) -> Tensor:
        """Custom forward pass for VAE encoder with skip connections."""
        sample = self.conv_in(sample)
        skip_connections = []

        # Down blocks
        for down_block in self.down_blocks:
            skip_connections.append(sample)
            sample = down_block(sample)

        # Middle block and final processing
        sample = self.mid_block(sample)
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        self.current_down_blocks = skip_connections
        return sample

    @staticmethod
    def _vae_decoder_forward(self, sample: Tensor,
                             latent_embeds: Optional[Tensor] = None) -> Tensor:
        """Custom forward pass for VAE decoder with skip connections."""
        sample = self.conv_in(sample)
        dtype = next(iter(self.up_blocks.parameters())).dtype
        sample = self.mid_block(sample, latent_embeds)
        sample = sample.to(dtype)

        if not self.ignore_skip:
            skip_convs = [
                self.skip_conv_1, self.skip_conv_2,
                self.skip_conv_3, self.skip_conv_4
            ]
            for idx, up_block in enumerate(self.up_blocks):
                skip_connection = skip_convs[idx](
                    self.incoming_skip_acts[::-1][idx] * self.gamma
                )
                sample = sample + skip_connection
                sample = up_block(sample, latent_embeds)
        else:
            for up_block in self.up_blocks:
                sample = up_block(sample, latent_embeds)

        # Final processing
        if latent_embeds is not None:
            norm_out = self.conv_norm_out(sample, latent_embeds)
        else:
            norm_out = self.conv_norm_out(sample)
        return self.conv_out(self.conv_act(norm_out))

    def set_eval(self):
        """Set model to evaluation mode."""
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def set_train(self):
        """Set model to training mode and enable gradient computation for LoRA parameters."""
        self.unet.train()
        self.vae.train()

        # Enable gradients for LoRA parameters
        for name, param in self.unet.named_parameters():
            param.requires_grad = "lora" in name
        self.unet.conv_in.requires_grad_(True)

        for name, param in self.vae.named_parameters():
            param.requires_grad = "lora" in name

        # Enable gradients for skip connection convolutions
        for i in range(1, 5):
            getattr(self.vae.decoder, f"skip_conv_{i}").requires_grad_(True)

    def forward(
            self,
            control_image: Tensor,
            prompt: Optional[str] = None,
            prompt_tokens: Optional[Tensor] = None,
            deterministic: bool = True,
            interpolation_factor: float = 1.0,
            noise_map: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass of the Pix2Pix-Turbo model.

        Args:
            control_image: Input control image
            prompt: Text prompt (mutually exclusive with prompt_tokens)
            prompt_tokens: Preprocessed prompt tokens (mutually exclusive with prompt)
            deterministic: Whether to use deterministic generation
            interpolation_factor: Interpolation factor between input and noise (0-1)
            noise_map: Optional noise map for non-deterministic generation

        Returns:
            Generated image tensor
        """
        # Validate inputs
        if (prompt is None) == (prompt_tokens is None):
            raise ValueError("Exactly one of prompt or prompt_tokens must be provided")

        # Process text prompt
        if prompt is not None:
            tokens = self.tokenizer(
                prompt,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(self.device)
            text_embeddings = self.text_encoder(tokens)[0]
        else:
            text_embeddings = self.text_encoder(prompt_tokens)[0]

        # Encode control image
        encoded_control = self.vae.encode(control_image).latent_dist.sample()
        encoded_control = encoded_control * self.vae.config.scaling_factor

        if deterministic:
            return self._deterministic_forward(encoded_control, text_embeddings)
        else:
            return self._stochastic_forward(
                encoded_control,
                text_embeddings,
                interpolation_factor,
                noise_map
            )

    def _deterministic_forward(
            self,
            encoded_control: Tensor,
            text_embeddings: Tensor
    ) -> Tensor:
        """Deterministic forward pass."""
        # U-Net prediction
        model_pred = self.unet(
            encoded_control,
            self.timesteps,
            encoder_hidden_states=text_embeddings
        ).sample

        # Denoising
        x_denoised = self.scheduler.step(
            model_pred,
            self.timesteps,
            encoded_control
        ).prev_sample.to(model_pred.dtype)

        # Decode
        self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
        decoded = self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample
        return torch.clamp(decoded, -1, 1)

    def _stochastic_forward(
            self,
            encoded_control: Tensor,
            text_embeddings: Tensor,
            interpolation_factor: float,
            noise_map: Tensor
    ) -> Tensor:
        """Stochastic forward pass with interpolation."""
        # Set interpolation weights
        self.unet.set_adapters(["default"], weights=[interpolation_factor])
        set_weights_and_activate_adapters(
            self.vae,
            ["vae_skip"],
            [interpolation_factor]
        )

        # Interpolate between input and noise
        unet_input = (
                encoded_control * interpolation_factor +
                noise_map * (1 - interpolation_factor)
        )

        # U-Net prediction with interpolation
        self.unet.conv_in.interpolation_factor = interpolation_factor
        unet_output = self.unet(
            unet_input,
            self.timesteps,
            encoder_hidden_states=text_embeddings
        ).sample
        self.unet.conv_in.interpolation_factor = None

        # Denoising
        x_denoised = self.scheduler.step(
            unet_output,
            self.timesteps,
            unet_input
        ).prev_sample.to(unet_output.dtype)

        # Decode with skip connections
        self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
        self.vae.decoder.gamma = interpolation_factor
        decoded = self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample
        return torch.clamp(decoded, -1, 1)

    def save_lora_weights(self, path: str):
        """Save LoRA weights and configurations."""
        state_dict = {
            "unet_lora_target_modules": self.target_modules["unet"],
            "vae_lora_target_modules": self.target_modules["vae"],
            "rank_unet": self.lora_ranks["unet"],
            "rank_vae": self.lora_ranks["vae"],
            "state_dict_unet": {
                k: v for k, v in self.unet.state_dict().items()
                if "lora" in k or "conv_in" in k
            },
            "state_dict_vae": {
                k: v for k, v in self.vae.state_dict().items()
                if "lora" in k or "skip" in k
            }
        }
        torch.save(state_dict, path)
