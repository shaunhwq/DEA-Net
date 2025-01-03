import os
from typing import Tuple

import torch
import cv2
import numpy as np
import torch.nn.functional as F

from code.model.backbone import Backbone


def pre_process(image: np.array, device: str, patch_size: int = 4):
    """
    :param image: Input image to transform to the model input
    :param device: Device to send input to
    :returns: Tensor input to model, in the shape [b, c, h, w]
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = np.ascontiguousarray(image_rgb.astype(np.float32)) / 255.0
    image_tensor = torch.from_numpy(image_rgb.transpose(2, 0, 1))
    image_tensor = image_tensor.unsqueeze(0).to(device)

    _, _, h, w = x.size()
    mod_pad_h = (patch_size - h % patch_size) % patch_size
    mod_pad_w = (patch_size - w % patch_size) % patch_size
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return image_tensor


def post_process(model_output: torch.Tensor, input_hw: Tuple[int, int]):
    """
    :param model_output: Output tensor produced by the model [b, c, h, w]
    :param input_hw: Tuple containing input image height and width
    :returns: Output image which can be displayed by OpenCV
    """
    h, w = input_hw
    model_output = model_output[:,:,:h,:w]

    image_rgb = model_output.squeeze(0).cpu().permute(1, 2, 0).numpy()
    image_rgb = (image_rgb * 255 + 0.5).clip(0, 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image_bgr


if __name__ == "__main__":
    video_path = "/Users/shaun/datasets/image_enhancement/dehaze/DVD/DrivingHazy/29_hazy_video.mp4"
    device = "cpu"
    weights_path = "trained_models/HAZE4K/PSNR3426_SSIM9885.pth"
    #weights_path = "trained_models/ITS/PSNR4131_SSIM9945.pth"
    #weights_path = "trained_models/OTS/PSNR3659_SSIM9897.pth"

    model = Backbone()
    weights = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(weights)
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(video_path)

    while True:
        frame_no = cap.get(cv2.CAP_PROP_POS_FRAMES)
        ret, frame = cap.read()

        in_tensor = pre_process(frame, device)
        with torch.no_grad():
            model_outputs = model(in_tensor)
        out_image = post_process(model_outputs, frame.shape[:2])

        display_image = np.vstack([frame, out_image])

        cv2.imshow("output", display_image)
        key = cv2.waitKey(1)
        if key & 255 == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
