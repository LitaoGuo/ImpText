import os
import shutil

import cv2
import numpy as np

from .image_robustness_toolkit import ImageRobustnessToolkit


class ToolsWrapper:
    """Named image enhancement tools used by tool-augmented baselines."""

    def __init__(self) -> None:
        self.toolkit = ImageRobustnessToolkit()

    def _custom_anisotropic_stretch(self, image_path: str, output_path: str) -> str:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        h, w = img.shape[:2]
        img_h = cv2.resize(img, (w * 2, h), interpolation=cv2.INTER_LINEAR)
        img_v = cv2.resize(img, (w, h * 2), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((h * 3, w * 2, 3), 255, dtype=np.uint8)
        canvas[0:h, 0 : w * 2] = img_h
        canvas[h : h * 3, 0:w] = img_v
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cv2.imwrite(output_path, canvas)
        return output_path

    def apply_tool(self, tool_name: str, image_path: str, output_path: str) -> str:
        if tool_name == "original":
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            shutil.copyfile(image_path, output_path)
            return output_path
        if tool_name == "channel_extraction":
            return self.toolkit.channel_extraction(image_path, channel="s", output_path=output_path)
        if tool_name.startswith("channel_extraction_"):
            return self.toolkit.channel_extraction(image_path, channel=tool_name.rsplit("_", 1)[-1], output_path=output_path)

        tool_map = {
            "adaptive_thresholding": self.toolkit.adaptive_thresholding,
            "canny_edge_extraction": self.toolkit.canny_edge_extraction,
            "jpeg_purify": self.toolkit.jpeg_purify,
            "posterization": self.toolkit.posterization,
            "sharpening": self.toolkit.sharpening,
            "anisotropic_stretch": self._custom_anisotropic_stretch,
            "clahe": self.toolkit.clahe,
            "downscale": self.toolkit.downscale_2x,
            "downscale_2x": self.toolkit.downscale_2x,
            "downscale_4x": self.toolkit.downscale_4x,
            "morphological_closing": self.toolkit.morphological_closing,
            "blackhat_extraction": self.toolkit.blackhat_extraction,
        }
        if tool_name not in tool_map:
            raise ValueError(f"Unknown tool: {tool_name}")
        return tool_map[tool_name](image_path, output_path=output_path)

    @staticmethod
    def get_all_tools() -> list[str]:
        return [
            "original",
            "adaptive_thresholding",
            "canny_edge_extraction",
            "channel_extraction_r",
            "channel_extraction_g",
            "channel_extraction_b",
            "channel_extraction_s",
            "jpeg_purify",
            "posterization",
            "sharpening",
            "anisotropic_stretch",
            "clahe",
            "downscale_2x",
            "downscale_4x",
            "morphological_closing",
            "blackhat_extraction",
        ]

