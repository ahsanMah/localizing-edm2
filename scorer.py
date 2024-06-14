import pickle

import numpy as np
import PIL.Image
import torch

import dnnlib


class EDMScorer(torch.nn.Module):
    def __init__(
        self,
        net,
        stop_ratio=0.8,  # Maximum ratio of noise levels to compute
        num_steps=10,  # Number of noise levels to evaluate.
        use_fp16=False,  # Execute the underlying model at FP16 precision?
        sigma_min=0.002,  # Minimum supported noise level.
        sigma_max=80,  # Maximum supported noise level.
        sigma_data=0.5,  # Expected standard deviation of the training data.
        rho=7,  # Time step discretization.
        device=torch.device("cpu"),  # Device to use.
    ):
        super().__init__()
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.net = net.eval()

        # Adjust noise levels based on how far we want to accumulate
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max * stop_ratio

        step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        print(t_steps)

        self.register_buffer(
            "sigma_steps", t_steps.to(torch.float64)
        )

    @torch.inference_mode()
    def forward(
        self,
        x,
        force_fp32=False,
    ):
        x = x.to(torch.float32)

        batch_scores = []
        for sigma in self.sigma_steps:
            xhat = self.net(x, sigma, force_fp32=force_fp32)
            c_skip = self.net.sigma_data**2 / (sigma**2 + self.net.sigma_data**2)
            score = xhat - (c_skip * x)

            # score_norms = score.mean(1)
            score_norms = score.square().sum(dim=(1, 2, 3)) ** 0.5
            batch_scores.append(score_norms)
        batch_scores = torch.stack(batch_scores, axis=1)

        return batch_scores


def test_runner():
    model_root = "https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions"
    netpath = f"{model_root}/edm2-img64-s-1073741-0.075.pkl"
    with dnnlib.util.open_url(netpath, verbose=1) as f:
        data = pickle.load(f)
    net = data["ema"]

    f = "goldfish.JPEG"
    image = (PIL.Image.open(f)).resize((64, 64), PIL.Image.Resampling.LANCZOS)
    image = np.array(image)
    image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
    x = torch.from_numpy(image).unsqueeze(0)
    model = EDMScorer(net, num_steps=5)
    scores = model(x)
    return scores

if __name__ == "__main__":
    s = test_runner()
    print(s)
