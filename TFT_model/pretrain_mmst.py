"""MMST-style self-supervised pretraining for the TFT remote-sensing branch.

Data contract matches ``MMST-ViT/data/pretrain_2017_2020.json``:
all available states, 2017-2020, six monthly Sentinel-2 dates and 28 daily
weather records per date. Weather tokens are Q; PVT patch tokens are K/V.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from models import VariableSelectionNetwork, WeatherRemotePatchPretrain
from pvt import PVTTinyEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MMST_ROOT = PROJECT_ROOT / "MMST-ViT"
sys.path.insert(0, str(MMST_ROOT))
from current_data import CurrentMMSTPretrainDataset


class MMSTWeatherPretrain(torch.nn.Module):
    def __init__(self, hidden_size=128, num_heads=4):
        super().__init__()
        self.pvt = PVTTinyEncoder(out_dim=hidden_size, drop=0.1)
        self.feature_projection = torch.nn.ModuleList(
            [torch.nn.Linear(1, hidden_size) for _ in range(9)]
        )
        self.vsn = VariableSelectionNetwork(
            {str(i): hidden_size for i in range(9)}, hidden_size, dropout=0.1
        )
        self.cross_modal = WeatherRemotePatchPretrain(hidden_size, num_heads, dropout=0.1)

    def forward(self, images, weather):
        # images: (B,3,H,W), weather: (B,28,9)
        patches = self.pvt.forward_tokens(images, include_cls=False)
        projected = {
            str(i): self.feature_projection[i](weather[..., i:i + 1])
            for i in range(9)
        }
        weather_tokens, _ = self.vsn(projected, torch.full(
            (weather.shape[0],), weather.shape[1], device=weather.device, dtype=torch.long
        ))
        return self.cross_modal(weather_tokens, patches)


def nt_xent(first, second, temperature=0.5):
    first = F.normalize(first, dim=-1)
    second = F.normalize(second, dim=-1)
    representations = torch.cat([first, second], dim=0)
    similarity = representations @ representations.T / temperature
    mask = ~torch.eye(similarity.shape[0], device=similarity.device, dtype=torch.bool)
    positive = torch.cat([torch.diag(similarity, first.shape[0]), torch.diag(similarity, -first.shape[0])])
    logits = similarity.masked_fill(~mask, -1e9)
    labels = torch.arange(similarity.shape[0], device=similarity.device)
    labels = (labels + first.shape[0]) % similarity.shape[0]
    return F.cross_entropy(logits, labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="../MMST-ViT/data/pretrain_2017_2020.json")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="../MMST-ViT/output_current/tft_pretrain.pth")
    args = parser.parse_args()
    records = json.loads(Path(args.manifest).resolve().read_text())
    states = {str(record["state"]).lower() for record in records}
    if len(states) < 6:
        raise ValueError("TFT pretraining must use the all-state MMST pretraining manifest")
    dataset = CurrentMMSTPretrainDataset(records, train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    model = MMSTWeatherPretrain().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    augment = transforms.Compose([
        transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        transforms.RandomGrayscale(p=0.2), transforms.GaussianBlur(9),
    ])
    for epoch in range(args.epochs):
        model.train(); total = 0.0; steps = 0
        for images, weather in loader:
            images = images[0].permute(0, 1, 4, 2, 3).contiguous().float() / 255.0
            weather = weather[0]
            t, g, _, _, _ = images.shape
            flat_images = images.reshape(t * g, 3, 224, 224).to(args.device)
            flat_weather = weather.reshape(t * g, 28, 9).to(args.device)
            for start in range(0, flat_images.shape[0], args.batch_size):
                image_batch = flat_images[start:start + args.batch_size]
                weather_batch = flat_weather[start:start + args.batch_size]
                if image_batch.shape[0] < 2: continue
                first = augment(image_batch); second = augment(image_batch)
                z1, _ = model(first, weather_batch); z2, _ = model(second, weather_batch)
                loss = nt_xent(z1, z2)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total += float(loss.item()); steps += 1
        print(f"epoch={epoch + 1} loss={total / max(steps, 1):.6f}")
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "epoch": epoch}, output)


if __name__ == "__main__":
    main()
