from __future__ import annotations

from threading import Event
from typing import Callable

import numpy as np


MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


class DepthEstimationCancelled(Exception):
	pass


class DepthEstimator:
	def __init__(self) -> None:
		self._processor = None
		self._model = None
		self._torch = None
		self._device = None

	@property
	def is_loaded(self) -> bool:
		return self._model is not None

	@property
	def device_name(self) -> str:
		return str(self._device) if self._device is not None else "미로드"

	def _load_model(self, progress: Callable[[str], None]) -> None:
		if self.is_loaded:
			return
		progress("깊이 모델과 프로세서를 준비합니다. 최초 실행 시 모델 다운로드가 필요합니다.")
		try:
			import torch
			from transformers import AutoImageProcessor, AutoModelForDepthEstimation
		except ImportError as error:
			raise RuntimeError(
				"깊이 기능 의존성이 없습니다. `python -m pip install -r requirements-depth.txt`를 실행하세요."
			) from error

		self._torch = torch
		self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self._processor = AutoImageProcessor.from_pretrained(MODEL_ID)
		self._model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID)
		self._model.to(self._device)
		self._model.eval()
		progress(f"깊이 모델 준비 완료 ({self._device})")

	def estimate_depth(
		self,
		rgb_image: np.ndarray,
		cancel_event: Event,
		progress: Callable[[str], None],
	) -> np.ndarray:
		if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
			raise ValueError("깊이 추정 입력은 RGB 이미지여야 합니다.")
		if cancel_event.is_set():
			raise DepthEstimationCancelled

		self._load_model(progress)
		if cancel_event.is_set():
			raise DepthEstimationCancelled

		from PIL import Image
		import torch.nn.functional as functional

		progress("사진의 상대 깊이를 분석합니다.")
		pil_image = Image.fromarray(np.ascontiguousarray(rgb_image), mode="RGB")
		inputs = self._processor(images=pil_image, return_tensors="pt")
		inputs = inputs.to(self._device)
		with self._torch.inference_mode():
			outputs = self._model(**inputs)
			depth = functional.interpolate(
				outputs.predicted_depth.unsqueeze(1),
				size=(rgb_image.shape[0], rgb_image.shape[1]),
				mode="bicubic",
				align_corners=False,
			).squeeze(0).squeeze(0)

		if cancel_event.is_set():
			raise DepthEstimationCancelled
		depth_map = depth.float().cpu().numpy()
		finite = np.isfinite(depth_map)
		if not finite.any():
			return np.full(depth_map.shape, 0.5, dtype=np.float32)
		minimum = float(depth_map[finite].min())
		maximum = float(depth_map[finite].max())
		if maximum - minimum < 1e-8:
			return np.full(depth_map.shape, 0.5, dtype=np.float32)
		depth_map = np.nan_to_num(depth_map, nan=minimum, posinf=maximum, neginf=minimum)
		return np.clip((depth_map - minimum) / (maximum - minimum), 0.0, 1.0).astype(np.float32)