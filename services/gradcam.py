# services/gradcam.py
import torch
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.device = next(model.parameters()).device
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        # assume target_layer is a module instance (e.g., model.conv5 or model.bn5)
        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()

    def generate(self, input_tensor, target_class):
        """
        input_tensor: torch tensor shape (1, C, H, W) on same device
        target_class: int of class index
        returns: heatmap as numpy HxW (0-255)
        """
        self.model.zero_grad()
        output = self.model(input_tensor)  # forward
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        score = logits[0, target_class]
        score.backward(retain_graph=True)

        gradients = self.gradients  # [N, C, h, w]
        activations = self.activations  # [N, C, h, w]
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # global avg pool
        cam = torch.sum(weights * activations, dim=1)  # [N, h, w]
        cam = torch.relu(cam)
        cam = cam.cpu().numpy()[0]
        cam = cv2.resize(cam, (input_tensor.size(3), input_tensor.size(2)))
        cam -= cam.min()
        if cam.max() != 0:
            cam = cam / cam.max()
        heatmap = np.uint8(255 * cam)
        return heatmap
