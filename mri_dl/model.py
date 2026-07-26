from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from pathlib import Path


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualUNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 base_channels: int = 32) -> None:
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base_channels)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base_channels * 4, base_channels * 8)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, 2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)
        self.output_conv = nn.Conv2d(base_channels, out_channels, 1)

    @staticmethod
    def _match_size(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        th, tw = ref.shape[-2:]
        dh, dw = th - t.shape[-2], tw - t.shape[-1]
        if dh > 0 or dw > 0:
            t = F.pad(t, [max(dw//2,0), max(dw-dw//2,0), max(dh//2,0), max(dh-dh//2,0)])
        if t.shape[-2] > th:
            s = (t.shape[-2]-th)//2
            t = t[..., s:s+th, :]
        if t.shape[-1] > tw:
            s = (t.shape[-1]-tw)//2
            t = t[..., :, s:s+tw]
        return t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self._match_size(self.up3(b), e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._match_size(self.up2(d3), e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._match_size(self.up1(d2), e1), e1], dim=1))
        # Predict only the correction (target - undersampled input).  The
        # input is added back in mri_dl.inference.predict_tensor().
        return self.output_conv(d1)

    def visualize(
        self,
        sample_input_shape: tuple[int, int, int] = (1, 256, 256),
        graphical: bool = True,
        show: bool = False,
        save_path: str | Path | None = None,
    ) -> dict[str, object]:
        return visualize_model(
            self,
            sample_input_shape=sample_input_shape,
            graphical=graphical,
            show=show,
            save_path=save_path,
        )


def visualize_model(
    model: nn.Module,
    sample_input_shape: tuple[int, int, int] = (1, 256, 256),
    graphical: bool = True,
    show: bool = False,
    save_path: str | Path | None = None,
) -> dict[str, object]:
    """Dynamically visualize a model from an actual forward pass.

    The textual output is derived from live module execution and is not a
    hardcoded architecture description. If matplotlib is available, a simple
    graph view is also saved/shown.
    """
    if len(sample_input_shape) != 3:
        raise ValueError("sample_input_shape must be (channels, height, width).")

    first_param = next(model.parameters(), None)
    device = first_param.device if first_param is not None else torch.device("cpu")
    example = torch.randn((1, *sample_input_shape), device=device)
    was_training = model.training
    model.eval()

    def _shape_of(value: object) -> object:
        if isinstance(value, torch.Tensor):
            return tuple(value.shape)
        if isinstance(value, (list, tuple)):
            return [_shape_of(item) for item in value]
        return type(value).__name__

    def _module_label(name: str, module: nn.Module) -> str:
        label = name or module.__class__.__name__
        return f"{label} [{module.__class__.__name__}]"

    try:
        captured: list[dict[str, object]] = []
        hooks = []

        for module_name, module in model.named_modules():
            if module_name == "" or any(module.children()):
                continue

            def _make_hook(name: str):
                def _hook(_module: nn.Module, inputs: tuple[object, ...], output: object) -> None:
                    captured.append(
                        {
                            "name": name,
                            "type": _module.__class__.__name__,
                            "input": _shape_of(inputs[0]) if inputs else None,
                            "output": _shape_of(output),
                        }
                    )

                return _hook

            hooks.append(module.register_forward_hook(_make_hook(module_name)))

        with torch.no_grad():
            output = model(example)

        for hook in hooks:
            hook.remove()

        node_lines: list[str] = []
        for index, item in enumerate(captured, start=1):
            node_lines.append(
                f"{index:02d}. {_module_label(str(item['name']), model.get_submodule(str(item['name'])))} "
                f"| input={item['input']} -> output={item['output']}"
            )

        print("\n" + "=" * 80)
        print("MODEL EXECUTION TRACE")
        print("=" * 80)
        for line in node_lines:
            print(line)

        rendered_path: Path | None = None
        if graphical:
            try:
                import matplotlib.pyplot as plt
                labels = []
                for item in captured:
                    label = f"{item['name']}\n{item['type']}\n{item['output']}"
                    labels.append(label[:48])

                if labels:
                    fig_width = max(12, len(labels) * 2)
                    fig, ax = plt.subplots(figsize=(fig_width, 4))
                    ax.axis("off")
                    ax.set_xlim(-0.75, len(labels) - 0.25)
                    ax.set_ylim(-1.2, 1.2)
                    for i, label in enumerate(labels):
                        box = plt.Rectangle((i - 0.42, -0.35), 0.84, 0.7, facecolor="#e5e7eb", edgecolor="#374151", linewidth=1.2)
                        ax.add_patch(box)
                        ax.text(i, 0, label, ha="center", va="center", fontsize=8)
                        if i < len(labels) - 1:
                            ax.annotate("", xy=(i + 0.48, 0), xytext=(i + 0.42, 0), arrowprops=dict(arrowstyle="->", lw=1.2))

                    output_base = Path(save_path) if save_path is not None else Path(__file__).with_name("residual_unet_traced_graph.png")
                    if output_base.suffix.lower() != ".png":
                        output_base = output_base.with_suffix(".png")
                    fig.tight_layout()
                    fig.savefig(output_base, dpi=180, bbox_inches="tight")
                    if show:
                        plt.show()
                    plt.close(fig)
                    rendered_path = output_base
                    print(f"Saved graphical trace to: {rendered_path}")
            except ImportError:
                plt = None
                print("Graphical visualization skipped: matplotlib is not available.")

        return {"trace": node_lines, "graph_path": rendered_path, "output_shape": tuple(output.shape) if isinstance(output, torch.Tensor) else type(output).__name__}
    finally:
        for hook in locals().get("hooks", []):
            hook.remove()
        if was_training:
            model.train()

