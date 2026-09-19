"""Export Dueling Double DQN checkpoints to the competition ONNX contract.

Contract (HANDOFF §1.6):
  Input:  state,    shape [1, 8], float32
  Output: q_values, shape [1, 2], float32
  opset >= 17
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from environment import OBS_DIM

try:
    import onnxruntime as ort
except ImportError:
    ort = None


class DuelingDQN(nn.Module):
    """Must match main.DuelingDQN exactly."""

    def __init__(self, state_dim=OBS_DIM, action_dim=2):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
        )

    def forward(self, x):
        h = self.feature(x)
        v = self.value(h)
        a = self.advantage(h)
        return v + (a - a.mean(dim=1, keepdim=True))


def load_model(model_path: str) -> DuelingDQN:
    checkpoint = torch.load(model_path, map_location="cpu")
    model = DuelingDQN(state_dim=OBS_DIM, action_dim=2)

    if "q_state_dict" in checkpoint:
        state_dict = checkpoint["q_state_dict"]
    elif "local_state_dict" in checkpoint:
        state_dict = checkpoint["local_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    return model


def verify_onnx(onnx_path: str, model: DuelingDQN, tol: float = 1e-5) -> bool:
    if ort is None:
        print("onnxruntime not installed — skipping numerical verify")
        return True

    rng = np.random.RandomState(0)
    x_np = rng.randn(4, OBS_DIM).astype(np.float32)
    with torch.no_grad():
        torch_out = model(torch.from_numpy(x_np)).numpy()

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_out = session.run(None, {"state": x_np})[0]

    max_err = float(np.max(np.abs(torch_out - ort_out)))
    ok = max_err < tol
    print(f"ONNX vs PyTorch max abs error: {max_err:.2e} ({'OK' if ok else 'FAIL'})")
    return ok


def export_model(model_path: str, output_path: str | None = None, opset: int = 17) -> str:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found.")

    if output_path is None:
        output_path = model_path.replace(".pth", ".onnx")

    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Architecture: DuelingDQN, state_dim={OBS_DIM}, params={n_params:,}")

    dummy = torch.randn(1, OBS_DIM)
    print(f"Exporting to {output_path} (opset={opset})...")
    # dynamo=False: legacy exporter — avoids Windows console emoji crashes in
    # torch.onnx dynamo path and keeps dynamic_axes working as documented.
    torch.onnx.export(
        model,
        dummy,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["state"],
        output_names=["q_values"],
        dynamic_axes={"state": {0: "batch"}, "q_values": {0: "batch"}},
        dynamo=False,
    )

    size_kb = os.path.getsize(output_path) / 1024
    print(f"Exported {output_path} ({size_kb:.1f} KB)")

    if not verify_onnx(output_path, model):
        raise RuntimeError("ONNX verification failed — refusing to ship mismatched graph")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Dueling DQN to ONNX (8-D contract)")
    parser.add_argument("model_path", type=str, help="Path to .pth checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Output .onnx path")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    export_model(args.model_path, args.output, args.opset)
