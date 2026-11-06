from __future__ import annotations

import cv2
import numpy as np


MAX_VIEW_SHIFT_PIXELS = 32
VIEW_DIRECTIONS = (
	(0, -1),
	(1, 0),
	(0, 1),
	(-1, 0),
)


def _render_view(
	rgb_image: np.ndarray,
	depth_map: np.ndarray,
	separation: int,
	focus_plane: float,
	view_x: int,
	view_y: int,
) -> np.ndarray:
	height, width = depth_map.shape
	x_coordinates, y_coordinates = np.meshgrid(
	np.arange(width, dtype=np.int32),
	np.arange(height, dtype=np.int32),
	)
	disparity = (focus_plane - depth_map) * MAX_VIEW_SHIFT_PIXELS * separation / 100.0
	target_x = x_coordinates + np.rint(disparity * view_x).astype(np.int32)
	target_y = y_coordinates + np.rint(disparity * view_y).astype(np.int32)
	valid = (
		(target_x >= 0)
		& (target_x < width)
		& (target_y >= 0)
		& (target_y < height)
	)
	source_x = x_coordinates[valid]
	source_y = y_coordinates[valid]
	destination_x = target_x[valid]
	destination_y = target_y[valid]
	source_depth = depth_map[valid]
	destination_indices = destination_y * width + destination_x
	depth_buffer = np.full(height * width, -np.inf, dtype=np.float32)
	np.maximum.at(depth_buffer, destination_indices, source_depth)
	visible = source_depth >= depth_buffer[destination_indices] - 1e-6

	view = np.zeros_like(rgb_image)
	view[destination_y[visible], destination_x[visible]] = rgb_image[
		source_y[visible], source_x[visible]
	]
	holes = depth_buffer.reshape(height, width) == -np.inf
	if holes.any():
		view = cv2.inpaint(
			view,
			holes.astype(np.uint8) * 255,
			2,
			cv2.INPAINT_TELEA,
		)
	return view


def render_four_views(
	rgb_image: np.ndarray,
	depth_map: np.ndarray,
	separation: int,
	focus_plane: float,
	invert_depth: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
		raise ValueError("원본 이미지는 RGB 형식이어야 합니다.")
	if depth_map.shape != rgb_image.shape[:2]:
		raise ValueError("깊이 지도와 원본 이미지의 크기가 일치해야 합니다.")
	if not 0 <= separation <= 100:
		raise ValueError("시점 간격은 0에서 100 사이여야 합니다.")

	depth = np.nan_to_num(depth_map.astype(np.float32), nan=0.5, posinf=1.0, neginf=0.0)
	depth = np.clip(depth, 0.0, 1.0)
	if invert_depth:
		depth = 1.0 - depth
	focus = float(np.clip(focus_plane, 0.0, 1.0))
	views = [
		_render_view(rgb_image, depth, separation, focus, direction_x, direction_y)
		for direction_x, direction_y in VIEW_DIRECTIONS
	]
	return (
		views[0],
		cv2.rotate(views[1], cv2.ROTATE_90_CLOCKWISE),
		cv2.rotate(views[2], cv2.ROTATE_180),
		cv2.rotate(views[3], cv2.ROTATE_90_COUNTERCLOCKWISE),
	)


def depth_preview_images(
	depth_map: np.ndarray,
	invert_depth: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
	depth = np.clip(np.nan_to_num(depth_map.astype(np.float32), nan=0.5), 0.0, 1.0)
	if invert_depth:
		depth = 1.0 - depth
	grayscale = np.round(depth * 255).astype(np.uint8)
	grayscale_rgb = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2RGB)
	color_rgb = cv2.cvtColor(
		cv2.applyColorMap(grayscale, cv2.COLORMAP_TURBO),
		cv2.COLOR_BGR2RGB,
	)
	return grayscale_rgb, color_rgb


def create_layered_test_scene(
	width: int = 640,
	height: int = 480,
) -> tuple[np.ndarray, np.ndarray]:
	rgb_image = np.full((height, width, 3), (18, 22, 30), dtype=np.uint8)
	depth_map = np.full((height, width), 0.12, dtype=np.float32)

	triangle = np.array(
		[
			[width * 0.70, height * 0.28],
			[width * 0.90, height * 0.72],
			[width * 0.55, height * 0.72],
		],
		dtype=np.int32,
	)
	cv2.fillPoly(rgb_image, [triangle], (70, 180, 245))
	cv2.fillPoly(depth_map, [triangle], 0.30)

	box_left = int(width * 0.37)
	box_top = int(height * 0.38)
	box_right = int(width * 0.59)
	box_bottom = int(height * 0.82)
	cv2.rectangle(rgb_image, (box_left, box_top), (box_right, box_bottom), (65, 205, 95), -1)
	cv2.rectangle(depth_map, (box_left, box_top), (box_right, box_bottom), 0.60, -1)

	circle_center = (int(width * 0.22), int(height * 0.62))
	cv2.circle(rgb_image, circle_center, max(12, int(min(width, height) * 0.105)), (235, 90, 185), -1)
	cv2.circle(depth_map, circle_center, max(12, int(min(width, height) * 0.105)), 0.90, -1)
	return rgb_image, depth_map