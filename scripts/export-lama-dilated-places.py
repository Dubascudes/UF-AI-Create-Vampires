"""Exports LaMa-Dilated with Places2 (scene) weights to TFLite for the Android app.

Qualcomm's AI Hub LaMa-Dilated export uses the CelebA-HQ face checkpoint, which paints faces into
holes. This script builds the same network from the LaMa source with the authors' Places2
checkpoint and writes lama_dilated_places.tflite with Qualcomm's contract: NHWC image (RGB 0-1)
and 0/1 mask (both 1x512x512) in, image with only the masked area generated out.

Setup (in a scratch conda env, Python 3.12):
  pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu
  pip install omegaconf pyyaml numpy litert-torch ai-edge-litert kornia easydict pytorch_lightning
  git clone --depth 1 https://github.com/advimman/lama.git lama-src
  # LaMa_models.zip from the Google Drive folder linked in the LaMa README (6.5 GB):
  unzip LaMa_models.zip "LaMa_models/lama-places/lama-dilated/*"
  PYTHONPATH=lama-src python export-lama-dilated-places.py
Then: adb push lama_dilated_places.tflite /sdcard/Android/data/edu.ufl.absent/files/
"""
import torch, yaml, numpy as np, litert_torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from saicinpainting.training.modules import make_generator

CONFIG = "LaMa_models/lama-places/lama-dilated/config.yaml"
CHECKPOINT = "LaMa_models/lama-places/lama-dilated/models/best.ckpt"


class CroppedConvTranspose(torch.nn.Module):
    """ConvTranspose2d with padding p and output_padding op, rewritten as an unpadded transposed
    convolution plus a crop (TFLite cannot express output_padding). Same weights, same output."""
    def __init__(self, conv):
        super().__init__()
        self.weight, self.bias = conv.weight, conv.bias
        self.stride, self.dilation, self.groups = conv.stride, conv.dilation, conv.groups
        self.padding, self.output_padding = conv.padding, conv.output_padding
    def forward(self, x):
        full = F.conv_transpose2d(x, self.weight, self.bias, self.stride, 0, 0, self.groups, self.dilation)
        (ph, pw), (oh, ow) = self.padding, self.output_padding
        h = (x.shape[2] - 1) * self.stride[0] - 2 * ph + self.dilation[0] * (self.weight.shape[2] - 1) + oh + 1
        w = (x.shape[3] - 1) * self.stride[1] - 2 * pw + self.dilation[1] * (self.weight.shape[3] - 1) + ow + 1
        return full[:, :, ph:ph + h, pw:pw + w]


class ReflectPad(torch.nn.Module):
    """Reflect padding built from slices and concatenation (exactly equal to F.pad "reflect").
    The converter expands nn.ReflectionPad2d / padding_mode="reflect" into hundreds of
    TRANSPOSE/GATHER_ND ops, and that graph fails on the Hexagon NPU; slices and concat do not."""
    def __init__(self, padding):
        super().__init__()
        self.left, self.right, self.top, self.bottom = padding
    def forward(self, x):
        w = x.shape[3]
        x = torch.cat([x[:, :, :, i:i + 1] for i in range(self.left, 0, -1)] + [x]
                      + [x[:, :, :, w - 1 - i:w - i] for i in range(1, self.right + 1)], dim=3)
        h = x.shape[2]
        return torch.cat([x[:, :, i:i + 1] for i in range(self.top, 0, -1)] + [x]
                         + [x[:, :, h - 1 - i:h - i] for i in range(1, self.bottom + 1)], dim=2)


def replace_reflect_padding(module):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.ReflectionPad2d):
            setattr(module, name, ReflectPad(child.padding))
        elif isinstance(child, torch.nn.Conv2d) and child.padding_mode == "reflect":
            ph, pw = child.padding
            child.padding, child.padding_mode = (0, 0), "zeros"
            child._reversed_padding_repeated_twice = [0, 0, 0, 0]
            setattr(module, name, torch.nn.Sequential(ReflectPad((pw, pw, ph, ph)), child))
        else:
            replace_reflect_padding(child)


def replace_transposed(module):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.ConvTranspose2d) and any(child.output_padding):
            setattr(module, name, CroppedConvTranspose(child))
        else:
            replace_transposed(child)


class LamaDilatedNHWC(torch.nn.Module):
    """NHWC image (RGB 0-1) and 0/1 mask in; image with only the masked area generated out."""
    def __init__(self, generator):
        super().__init__()
        self.generator = generator
    def forward(self, image, mask):
        image = image.permute(0, 3, 1, 2)
        mask = mask.permute(0, 3, 1, 2)
        predicted = self.generator(torch.cat([image * (1 - mask), mask], dim=1))
        return (mask * predicted + (1 - mask) * image).permute(0, 2, 3, 1)


cfg = OmegaConf.create(yaml.safe_load(open(CONFIG)))
gen = make_generator(cfg, **cfg.generator)
gen.load_state_dict({k[len("generator."):]: v for k, v in torch.load(CHECKPOINT, map_location="cpu", weights_only=False)["state_dict"].items() if k.startswith("generator.")})
gen.eval()
image = torch.rand(1, 512, 512, 3)
mask = torch.zeros(1, 512, 512, 1); mask[:, 150:350, 200:300] = 1
with torch.no_grad():
    original = LamaDilatedNHWC(gen)(image, mask).numpy()
replace_transposed(gen)
replace_reflect_padding(gen)
model = LamaDilatedNHWC(gen).eval()
with torch.no_grad():
    rewritten = model(image, mask).numpy()
print("rewrite max |diff| =", float(np.abs(rewritten - original).max()))
with torch.no_grad():
    rewritten = model(image, mask).numpy()
print("padding rewrite max |diff| =", float(np.abs(rewritten - original).max()))
litert_torch.convert(model, (image, mask)).export("lama_dilated_places.tflite")

from ai_edge_litert.interpreter import Interpreter
it = Interpreter(model_path="lama_dilated_places.tflite", num_threads=8); it.allocate_tensors()
ins = it.get_input_details()
for d in ins: print("IN", d["name"], d["shape"])
print("OUT", it.get_output_details()[0]["shape"])
it.set_tensor(ins[0]["index"], image.numpy()); it.set_tensor(ins[1]["index"], mask.numpy()); it.invoke()
out = it.get_tensor(it.get_output_details()[0]["index"])
print("tflite max |diff| =", float(np.abs(out - original).max()))
