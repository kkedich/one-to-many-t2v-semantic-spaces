import lpips
import os
import torch
from DISTS_pytorch import DISTS

from text2video.metrics.utils import videos2images

# Set PYTORCH_CUDA_ALLOC_CONF environment variable
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class LPIPSMetric:
    def __init__(self, net="alex", device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = lpips.LPIPS(net=net).to(device)

    def __call__(self, y_true, y_pred):
        # Check if input is a batch of videos instead of images
        original_shape = y_true.shape
        if len(original_shape) > 4:
            # Convert to batch of images
            y_true = videos2images(y_true)
            y_pred = videos2images(y_pred)

        # Now, we need to convert the tensors to torch tensors as lpips is in pytorch
        with torch.no_grad():
            # Convert npy to torch tensors
            y_true_t = torch.from_numpy(y_true.numpy())
            y_pred_t = torch.from_numpy(y_pred.numpy())

            y_true_t = torch.permute(y_true_t, (0, 3, 1, 2)).to(self.device)
            y_pred_t = torch.permute(y_pred_t, (0, 3, 1, 2)).to(self.device)

            # From the original repo
            # Variables are PyTorch Tensor/Variable with shape Nx3xHxW
            # (N patches of size HxW, RGB images scaled in [-1,+1]).
            # This returns d, a length N Tensor/Variable.
            # loss_fn = lpips.LPIPS(net='alex')
            # distance is N array
            distance = self.model.forward(y_true_t, y_pred_t)

            if len(original_shape) > 4:
                # Reduce to number of videos
                distance = distance.view((original_shape[0], original_shape[1]))
                distance = torch.mean(distance, dim=-1)

        return distance.cpu().numpy()


class DISTSMetric:
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model = DISTS().to(device)

    def __call__(self, y_true, y_pred):
        # Check if input is a batch of videos instead of images
        original_shape = y_true.shape
        if len(original_shape) > 4:
            # Convert to batch of images
            y_true = videos2images(y_true)
            y_pred = videos2images(y_pred)

        with torch.no_grad():
            # Now, we need to convert the tensors to torch tensors as DISTS is in pytorch
            # Convert npy to torch tensors
            y_true_t = torch.from_numpy(y_true.numpy())
            y_pred_t = torch.from_numpy(y_pred.numpy())

            # calculate DISTS between X, Y (a batch of RGB images, data range: 0~1)
            # images (N,C,H,W)
            y_true_t = torch.permute(y_true_t, (0, 3, 1, 2)).to(self.device)
            y_pred_t = torch.permute(y_pred_t, (0, 3, 1, 2)).to(self.device)

            score = self.model(y_true_t, y_pred_t)

            if len(original_shape) > 4:
                # Reduce to number of videos
                score = score.view((original_shape[0], original_shape[1]))
                score = torch.mean(score, dim=-1)

        return score.cpu().numpy()
