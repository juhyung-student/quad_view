import logging
import sys
from functools import lru_cache
from pathlib import Path
from threading import Event

import cv2
import numpy as np
from PySide6.QtCore import QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import (
	QColor,
	QImage,
	QKeyEvent,
	QMovie,
	QPainter,
	QPen,
	QPixmap,
	QTransform,
)
from PySide6.QtWidgets import (
	QApplication,
	QCheckBox,
	QDialog,
	QFileDialog,
	QHBoxLayout,
	QLabel,
	QMainWindow,
	QMessageBox,
	QPushButton,
	QSlider,
	QSizePolicy,
	QStackedWidget,
	QVBoxLayout,
	QWidget,
)

from depth_estimator import DepthEstimationCancelled, DepthEstimator
from depth_renderer import (
	create_layered_test_scene,
	depth_preview_images,
	render_four_views,
)


logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s %(levelname)s %(message)s",
)
LOGGER = logging.getLogger("quad_view")

PHOTO_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
GIF_EXTENSIONS = {".gif"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
SUPPORTED_EXTENSIONS = PHOTO_EXTENSIONS | GIF_EXTENSIONS | VIDEO_EXTENSIONS
PANEL_GAP = 0
GROUP_SCALE = 0.80
CURVE_SLIDER_MAX = 100
MAX_CURVE_FRACTION = 0.15
STEREO_SLIDER_MAX = 100
DEFAULT_STEREO_SEPARATION = 30
DEFAULT_FOCUS_PLANE = 50
FILE_FILTER = (
	"미디어 파일 (*.png *.jpg *.jpeg *.webp *.gif *.mp4 *.avi *.mov *.mkv);;"
	"사진 (*.png *.jpg *.jpeg *.webp);;"
	"애니메이션 GIF (*.gif);;"
	"동영상 (*.mp4 *.avi *.mov *.mkv)"
)


def image_from_video_frame(frame: np.ndarray) -> QImage:
	"""Convert an OpenCV BGR frame into an owned RGB QImage."""
	rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	return image_from_rgb_array(rgb_frame)


def image_from_rgb_array(rgb_image: np.ndarray) -> QImage:
	height, width, _ = rgb_image.shape
	return QImage(
		rgb_image.data,
		width,
		height,
		rgb_image.strides[0],
		QImage.Format.Format_RGB888,
	).copy()


def qimage_to_rgb_array(image: QImage) -> np.ndarray:
	rgb_image = image.convertToFormat(QImage.Format.Format_RGB888)
	height = rgb_image.height()
	width = rgb_image.width()
	buffer = np.frombuffer(
		rgb_image.constBits(),
		dtype=np.uint8,
		count=rgb_image.sizeInBytes(),
	)
	return (
		buffer.reshape(height, rgb_image.bytesPerLine())[:, :width * 3]
		.reshape(height, width, 3)
		.copy()
	)


class DepthWorker(QThread):
	progress = Signal(str)
	depth_ready = Signal(int, object)
	failed = Signal(int, str)
	cancelled = Signal(int)

	def __init__(
		self,
		estimator: DepthEstimator,
		image: QImage,
		generation: int,
		parent=None,
	) -> None:
		super().__init__(parent)
		self._estimator = estimator
		self._image = image.copy()
		self._generation = generation
		self._cancel_event = Event()

	def cancel(self) -> None:
		self._cancel_event.set()

	def run(self) -> None:
		try:
			rgb_image = qimage_to_rgb_array(self._image)
			depth_map = self._estimator.estimate_depth(
				rgb_image,
				self._cancel_event,
				self.progress.emit,
			)
			if self._cancel_event.is_set():
				self.cancelled.emit(self._generation)
			else:
				self.depth_ready.emit(self._generation, depth_map)
		except DepthEstimationCancelled:
			self.cancelled.emit(self._generation)
		except Exception as error:
			LOGGER.exception("사진 깊이 추정에 실패했습니다.")
			self.failed.emit(self._generation, str(error))


def create_test_pattern(width: int = 640, height: int = 480) -> QImage:
	image = QImage(width, height, QImage.Format.Format_RGB32)
	image.fill(QColor("black"))
	painter = QPainter(image)
	painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
	painter.setPen(QPen(QColor("#666666"), 1))
	for x in range(0, width, 40):
		painter.drawLine(x, 0, x, height - 1)
	for y in range(0, height, 40):
		painter.drawLine(0, y, width - 1, y)
	painter.setPen(QPen(QColor("#ffffff"), 2))
	painter.drawLine(width // 2, 0, width // 2, height - 1)
	painter.drawLine(0, height // 2, width - 1, height // 2)
	painter.setPen(QPen(QColor("#ff3030"), 3))
	painter.drawLine(0, height - 8, width - 1, height - 8)
	corner_colors = (
		(QColor("#ff3030"), 0, 0, 1, 1),
		(QColor("#30ff30"), width - 1, 0, -1, 1),
		(QColor("#3090ff"), 0, height - 1, 1, -1),
		(QColor("#ffff30"), width - 1, height - 1, -1, -1),
	)
	for color, x, y, x_direction, y_direction in corner_colors:
		painter.setPen(QPen(color, 4))
		painter.drawLine(x, y, x + x_direction * 28, y)
		painter.drawLine(x, y, x, y + y_direction * 28)
	painter.end()
	return image


@lru_cache(maxsize=4)
def curve_remap_maps(
	width: int,
	height: int,
	strength: int,
	reversed_direction: bool,
) -> tuple[np.ndarray, np.ndarray]:
	x = np.linspace(0.0, 1.0, width, dtype=np.float32)
	center_curve = 4.0 * x * (1.0 - x)
	profile = 1.0 - center_curve if reversed_direction else center_curve
	amount = MAX_CURVE_FRACTION * strength / CURVE_SLIDER_MAX
	vertical_scale = 1.0 - amount * profile
	map_x = np.broadcast_to(x[np.newaxis, :] * (width - 1), (height, width)).copy()
	y = np.arange(height, dtype=np.float32)[:, np.newaxis]
	map_y = np.broadcast_to(y / vertical_scale[np.newaxis, :], (height, width)).copy()
	return map_x, map_y


def curve_correct_image(
	image: QImage,
	strength: int,
	reversed_direction: bool,
) -> QImage:
	if image.isNull() or strength <= 0:
		return image
	has_alpha = image.hasAlphaChannel()
	source_format = (
		QImage.Format.Format_RGBA8888
		if has_alpha
		else QImage.Format.Format_RGB888
	)
	source_image = image.convertToFormat(source_format)
	height = source_image.height()
	width = source_image.width()
	channel_count = 4 if has_alpha else 3
	buffer = np.frombuffer(
		source_image.constBits(),
		dtype=np.uint8,
		count=source_image.sizeInBytes(),
	)
	pixels = buffer.reshape(height, source_image.bytesPerLine())[
		:, :width * channel_count
	].reshape(height, width, channel_count)
	map_x, map_y = curve_remap_maps(width, height, strength, reversed_direction)
	corrected = cv2.remap(
		pixels,
		map_x,
		map_y,
		cv2.INTER_LINEAR,
		borderMode=cv2.BORDER_CONSTANT,
		borderValue=(0, 0, 0, 0) if has_alpha else (0, 0, 0),
	)
	return QImage(
		corrected.data,
		width,
		height,
		corrected.strides[0],
		source_format,
	).copy()


def quad_layout_rectangles(
	canvas_width: int,
	canvas_height: int,
	image_width: int,
	image_height: int,
) -> tuple[tuple[QRectF, ...], QRectF]:
	if min(canvas_width, canvas_height, image_width, image_height) <= 0:
		return (), QRectF()

	outer_size = image_width + 2 * image_height
	available_width = max(0, canvas_width - 2 * PANEL_GAP)
	available_height = max(0, canvas_height - 2 * PANEL_GAP)
	fit_scale = min(available_width / outer_size, available_height / outer_size)
	scale = fit_scale * GROUP_SCALE
	panel_width = image_width * scale
	panel_height = image_height * scale
	center_x = canvas_width / 2
	center_y = canvas_height / 2

	panels = (
		QRectF(
			center_x - panel_width / 2,
			center_y - panel_width / 2 - PANEL_GAP - panel_height,
			panel_width,
			panel_height,
		),
		QRectF(
			center_x + panel_width / 2 + PANEL_GAP,
			center_y - panel_width / 2,
			panel_height,
			panel_width,
		),
		QRectF(
			center_x - panel_width / 2,
			center_y + panel_width / 2 + PANEL_GAP,
			panel_width,
			panel_height,
		),
		QRectF(
			center_x - panel_width / 2 - PANEL_GAP - panel_height,
			center_y - panel_width / 2,
			panel_height,
			panel_width,
		),
	)
	center = QRectF(
		center_x - panel_width / 2,
		center_y - panel_width / 2,
		panel_width,
		panel_width,
	)
	return panels, center


class QuadCanvas(QWidget):
	def __init__(self, parent: QWidget | None = None) -> None:
		super().__init__(parent)
		self.setMinimumSize(240, 240)
		self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
		self._source_images: list[QImage] = []
		self._rotated_images: list[QImage] = []
		self._correction_enabled = False
		self._correction_strength = 0
		self._reversed_direction = False

	def set_image(self, image: QImage) -> None:
		self._source_images = [
			image,
			image.transformed(QTransform().rotate(90), Qt.TransformationMode.SmoothTransformation),
			image.transformed(QTransform().rotate(180), Qt.TransformationMode.SmoothTransformation),
			image.transformed(QTransform().rotate(270), Qt.TransformationMode.SmoothTransformation),
		]
		self._refresh_images()

	def set_view_images(self, images: tuple[QImage, QImage, QImage, QImage]) -> None:
		if len(images) != 4 or any(image.isNull() for image in images):
			raise ValueError("입체 출력에는 유효한 이미지 네 장이 필요합니다.")
		self._source_images = list(images)
		self._refresh_images()

	def set_correction(self, enabled: bool, strength: int, reversed_direction: bool) -> None:
		self._correction_enabled = enabled
		self._correction_strength = strength
		self._reversed_direction = reversed_direction
		self._refresh_images()

	def _refresh_images(self) -> None:
		if self._correction_enabled and self._correction_strength > 0:
			self._rotated_images = [
				curve_correct_image(image, self._correction_strength, self._reversed_direction)
				for image in self._source_images
			]
		else:
			self._rotated_images = self._source_images.copy()
		self.update()

	def clear(self) -> None:
		self._source_images.clear()
		self._rotated_images.clear()
		self.update()

	def paintEvent(self, event) -> None:
		painter = QPainter(self)
		painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
		painter.fillRect(self.rect(), Qt.GlobalColor.black)
		if not self._rotated_images or self._rotated_images[0].isNull():
			painter.end()
			return

		original = self._rotated_images[0]
		rectangles, _ = quad_layout_rectangles(
			self.width(),
			self.height(),
			original.width(),
			original.height(),
		)
		for image, rectangle in zip(self._rotated_images, rectangles):
			painter.drawImage(rectangle, image)
		painter.end()


class QuadViewWindow(QMainWindow):
	def __init__(self) -> None:
		super().__init__()
		self.setWindowTitle("쿼드 뷰")
		self.resize(1100, 800)
		self._video: cv2.VideoCapture | None = None
		self._movie: QMovie | None = None
		self._frame_interval_ms = 33
		self._is_video = False
		self._current_media_image = QImage()
		self._test_pattern = create_test_pattern()
		self._test_pattern_active = False
		self._depth_estimator = DepthEstimator()
		self._depth_worker: DepthWorker | None = None
		self._depth_generation = 0
		self._depth_map: np.ndarray | None = None
		self._depth_source_rgb: np.ndarray | None = None
		self._depth_cache: dict[tuple[str, int, int], np.ndarray] = {}
		self._current_depth_key: tuple[str, int, int] | None = None
		self._stereo_mode = False
		self._layered_scene_active = False

		self._timer = QTimer(self)
		self._timer.timeout.connect(self._advance_video)

		self._pages = QStackedWidget()
		self.setCentralWidget(self._pages)
		self._pages.addWidget(self._create_select_page())
		self._pages.addWidget(self._create_media_page())
		self._pages.setCurrentIndex(0)
		self.setStyleSheet("QMainWindow, QWidget { background: #000000; color: #ffffff; }")

	def _create_select_page(self) -> QWidget:
		page = QWidget()
		layout = QVBoxLayout(page)
		layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
		title = QLabel("사진 또는 동영상을 선택하세요")
		title.setAlignment(Qt.AlignmentFlag.AlignCenter)
		title.setStyleSheet("font-size: 22px; font-weight: 600;")
		formats = QLabel(
			"사진: PNG, JPG, JPEG, WEBP  |  애니메이션: GIF  |  "
			"동영상: MP4, AVI, MOV, MKV"
		)
		formats.setAlignment(Qt.AlignmentFlag.AlignCenter)
		button = QPushButton("파일 선택")
		button.setMinimumSize(180, 48)
		button.clicked.connect(self._choose_file)
		layout.addWidget(title)
		layout.addSpacing(20)
		layout.addWidget(formats)
		layout.addSpacing(12)
		layout.addWidget(button, alignment=Qt.AlignmentFlag.AlignHCenter)
		return page

	def _create_media_page(self) -> QWidget:
		page = QWidget()
		self._media_layout = QVBoxLayout(page)
		self._media_layout.setContentsMargins(12, 8, 12, 10)
		self._media_layout.setSpacing(6)

		self._toolbar = QWidget()
		toolbar = QHBoxLayout(self._toolbar)
		toolbar.setContentsMargins(0, 0, 0, 0)
		back_button = QPushButton("뒤로 가기")
		back_button.clicked.connect(self._return_to_select)
		choose_again_button = QPushButton("다른 파일 선택")
		choose_again_button.clicked.connect(self._choose_file)
		fullscreen_button = QPushButton("전체 화면")
		fullscreen_button.clicked.connect(self._toggle_full_screen)
		self._test_pattern_button = QPushButton("보정 테스트 패턴")
		self._test_pattern_button.clicked.connect(self._show_test_pattern)
		self._media_button = QPushButton("미디어 보기")
		self._media_button.clicked.connect(self._show_media)
		self._media_button.hide()
		toolbar.addWidget(back_button)
		toolbar.addWidget(choose_again_button)
		toolbar.addWidget(self._test_pattern_button)
		toolbar.addWidget(self._media_button)
		toolbar.addStretch(1)
		toolbar.addWidget(fullscreen_button)
		self._media_layout.addWidget(self._toolbar)

		self._canvas = QuadCanvas()
		self._media_layout.addWidget(self._canvas, stretch=1)

		self._playback = QWidget()
		playback_layout = QHBoxLayout(self._playback)
		playback_layout.setContentsMargins(0, 0, 0, 0)
		playback_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
		play_button = QPushButton("재생")
		pause_button = QPushButton("일시정지")
		restart_button = QPushButton("처음부터")
		play_button.clicked.connect(self._play_media)
		pause_button.clicked.connect(self._pause_media)
		restart_button.clicked.connect(self._restart_media)
		playback_layout.addWidget(play_button)
		playback_layout.addWidget(pause_button)
		playback_layout.addWidget(restart_button)
		self._media_layout.addWidget(self._playback)
		self._playback.hide()

		self._correction_controls = QWidget()
		correction_layout = QHBoxLayout(self._correction_controls)
		correction_layout.setContentsMargins(0, 0, 0, 0)
		self._correction_checkbox = QCheckBox("투영 보정")
		self._reverse_checkbox = QCheckBox("보정 방향 반전")
		self._correction_slider = QSlider(Qt.Orientation.Horizontal)
		self._correction_slider.setRange(0, CURVE_SLIDER_MAX)
		self._correction_slider.setValue(0)
		self._correction_value = QLabel("0")
		self._correction_checkbox.toggled.connect(self._update_correction)
		self._reverse_checkbox.toggled.connect(self._update_correction)
		self._correction_slider.valueChanged.connect(self._update_correction)
		correction_layout.addWidget(self._correction_checkbox)
		correction_layout.addWidget(self._correction_slider, stretch=1)
		correction_layout.addWidget(self._correction_value)
		correction_layout.addWidget(self._reverse_checkbox)
		self._media_layout.addWidget(self._correction_controls)

		self._depth_controls = QWidget()
		depth_layout = QVBoxLayout(self._depth_controls)
		depth_layout.setContentsMargins(0, 0, 0, 0)
		depth_buttons = QHBoxLayout()
		self._depth_button = QPushButton("입체 효과 켜기")
		self._depth_button.clicked.connect(self._start_depth_estimation)
		self._original_mode_button = QPushButton("원본 모드로 돌아가기")
		self._original_mode_button.clicked.connect(self._show_original_mode)
		self._original_mode_button.hide()
		self._depth_preview_button = QPushButton("깊이 지도 미리보기")
		self._depth_preview_button.clicked.connect(self._show_depth_preview)
		self._depth_preview_button.setEnabled(False)
		self._layered_scene_button = QPushButton("합성 입체 장면")
		self._layered_scene_button.clicked.connect(self._show_layered_scene)
		self._cancel_depth_button = QPushButton("분석 취소")
		self._cancel_depth_button.clicked.connect(self._cancel_depth_estimation)
		self._cancel_depth_button.hide()
		self._depth_invert_checkbox = QCheckBox("깊이 반전")
		self._depth_invert_checkbox.toggled.connect(self._depth_parameters_changed)
		depth_buttons.addWidget(self._depth_button)
		depth_buttons.addWidget(self._original_mode_button)
		depth_buttons.addWidget(self._depth_preview_button)
		depth_buttons.addWidget(self._layered_scene_button)
		depth_buttons.addWidget(self._cancel_depth_button)
		depth_buttons.addWidget(self._depth_invert_checkbox)
		depth_layout.addLayout(depth_buttons)

		stereo_sliders = QHBoxLayout()
		self._separation_label = QLabel("시점 간격")
		self._separation_slider = QSlider(Qt.Orientation.Horizontal)
		self._separation_slider.setRange(0, STEREO_SLIDER_MAX)
		self._separation_slider.setValue(DEFAULT_STEREO_SEPARATION)
		self._separation_slider.setEnabled(False)
		self._separation_value = QLabel(str(DEFAULT_STEREO_SEPARATION))
		self._focus_label = QLabel("초점면")
		self._focus_slider = QSlider(Qt.Orientation.Horizontal)
		self._focus_slider.setRange(0, STEREO_SLIDER_MAX)
		self._focus_slider.setValue(DEFAULT_FOCUS_PLANE)
		self._focus_slider.setEnabled(False)
		self._focus_value = QLabel(str(DEFAULT_FOCUS_PLANE))
		self._separation_slider.valueChanged.connect(self._depth_parameters_changed)
		self._focus_slider.valueChanged.connect(self._depth_parameters_changed)
		stereo_sliders.addWidget(self._separation_label)
		stereo_sliders.addWidget(self._separation_slider, stretch=1)
		stereo_sliders.addWidget(self._separation_value)
		stereo_sliders.addWidget(self._focus_label)
		stereo_sliders.addWidget(self._focus_slider, stretch=1)
		stereo_sliders.addWidget(self._focus_value)
		depth_layout.addLayout(stereo_sliders)
		self._depth_status = QLabel("상대 깊이 분석은 사진에서 사용할 수 있습니다.")
		depth_layout.addWidget(self._depth_status)
		self._media_layout.addWidget(self._depth_controls)
		self._depth_controls.hide()
		return page

	def _update_correction(self, value: int = 0) -> None:
		strength = self._correction_slider.value()
		self._correction_value.setText(str(strength))
		self._canvas.set_correction(
			self._correction_checkbox.isChecked(),
			strength,
			self._reverse_checkbox.isChecked(),
		)

	def _start_depth_estimation(self) -> None:
		if self._is_playback_media() or self._current_media_image.isNull():
			QMessageBox.information(self, "입체 효과", "입체 효과는 사진에서 사용할 수 있습니다.")
			return
		if self._depth_worker is not None:
			return
		if (
			self._depth_map is not None
			and self._depth_source_rgb is not None
			and not self._layered_scene_active
		):
			self._stereo_mode = True
			self._layered_scene_active = False
			self._original_mode_button.show()
			self._render_stereo_views()
			return

		worker = DepthWorker(
			self._depth_estimator,
			self._current_media_image,
			self._depth_generation,
			self,
		)
		self._depth_worker = worker
		worker.progress.connect(self._depth_status.setText)
		worker.depth_ready.connect(self._on_depth_ready)
		worker.failed.connect(self._on_depth_failed)
		worker.cancelled.connect(self._on_depth_cancelled)
		worker.finished.connect(lambda current=worker: self._on_depth_worker_finished(current))
		worker.finished.connect(worker.deleteLater)
		self._depth_button.setEnabled(False)
		self._cancel_depth_button.show()
		self._depth_status.setText("깊이 분석을 준비합니다.")
		worker.start()

	def _cancel_depth_estimation(self) -> None:
		if self._depth_worker is not None and self._depth_worker.isRunning():
			self._depth_worker.cancel()
			self._cancel_depth_button.setEnabled(False)
			self._depth_status.setText("분석 취소를 요청했습니다.")

	def _on_depth_worker_finished(self, worker: DepthWorker) -> None:
		if self._depth_worker is worker:
			self._depth_worker = None
			self._depth_button.setEnabled(not self._is_playback_media())
			self._cancel_depth_button.setEnabled(True)
			self._cancel_depth_button.hide()

	def _on_depth_ready(self, generation: int, depth_map: np.ndarray) -> None:
		if generation != self._depth_generation or self._current_media_image.isNull():
			return
		self._depth_map = depth_map
		self._depth_source_rgb = qimage_to_rgb_array(self._current_media_image)
		if self._current_depth_key is not None:
			self._depth_cache[self._current_depth_key] = depth_map
			while len(self._depth_cache) > 3:
				self._depth_cache.pop(next(iter(self._depth_cache)))
		self._layered_scene_active = False
		self._depth_preview_button.setEnabled(True)
		self._separation_slider.setEnabled(True)
		self._focus_slider.setEnabled(True)
		self._depth_status.setText(
			f"상대 깊이 분석 완료 ({self._depth_estimator.device_name}). "
			"깊이값은 실제 거리 단위가 아닙니다."
		)
		self._stereo_mode = True
		self._original_mode_button.show()
		self._render_stereo_views()

	def _on_depth_failed(self, generation: int, message: str) -> None:
		if generation != self._depth_generation:
			return
		self._depth_status.setText("깊이 분석에 실패했습니다.")
		QMessageBox.critical(self, "깊이 분석 실패", message)

	def _on_depth_cancelled(self, generation: int) -> None:
		if generation == self._depth_generation:
			self._depth_status.setText("깊이 분석을 취소했습니다.")

	def _invalidate_depth_result(self) -> None:
		self._depth_generation += 1
		if self._depth_worker is not None and self._depth_worker.isRunning():
			self._depth_worker.cancel()
		self._depth_map = None
		self._depth_source_rgb = None
		self._current_depth_key = None
		self._stereo_mode = False
		self._layered_scene_active = False
		self._original_mode_button.hide()
		self._depth_preview_button.setEnabled(False)
		self._separation_slider.setEnabled(False)
		self._focus_slider.setEnabled(False)

	def _depth_parameters_changed(self, value: int = 0) -> None:
		separation = self._separation_slider.value()
		focus = self._focus_slider.value()
		self._separation_value.setText(str(separation))
		self._focus_value.setText(str(focus))
		if self._stereo_mode and self._depth_map is not None:
			self._render_stereo_views()

	def _render_stereo_views(self) -> None:
		if self._depth_map is None or self._depth_source_rgb is None:
			return
		try:
			views = render_four_views(
				self._depth_source_rgb,
				self._depth_map,
				self._separation_slider.value(),
				self._focus_slider.value() / STEREO_SLIDER_MAX,
				self._depth_invert_checkbox.isChecked(),
			)
			self._canvas.set_view_images(tuple(image_from_rgb_array(view) for view in views))
		except (ValueError, cv2.error) as error:
			LOGGER.exception("깊이 기반 시점 생성에 실패했습니다.")
			QMessageBox.critical(self, "입체 영상 생성 실패", str(error))

	def _show_original_mode(self) -> None:
		self._stereo_mode = False
		if self._layered_scene_active:
			self._depth_map = None
			self._depth_source_rgb = None
			self._layered_scene_active = False
			self._depth_preview_button.setEnabled(False)
			self._separation_slider.setEnabled(False)
			self._focus_slider.setEnabled(False)
			self._depth_status.setText("원본 사진 모드입니다.")
		self._original_mode_button.hide()
		if not self._current_media_image.isNull():
			self._canvas.set_image(self._current_media_image)

	def _show_layered_scene(self) -> None:
		if self._is_playback_media():
			return
		if self._depth_worker is not None and self._depth_worker.isRunning():
			self._depth_generation += 1
			self._depth_worker.cancel()
		rgb_image, depth_map = create_layered_test_scene()
		self._depth_source_rgb = rgb_image
		self._depth_map = depth_map
		self._current_depth_key = None
		self._layered_scene_active = True
		self._stereo_mode = True
		self._depth_preview_button.setEnabled(True)
		self._separation_slider.setEnabled(True)
		self._focus_slider.setEnabled(True)
		self._original_mode_button.show()
		self._depth_status.setText("합성 장면: 원, 사각형, 삼각형에 서로 다른 깊이를 지정했습니다.")
		self._render_stereo_views()

	def _show_depth_preview(self) -> None:
		if self._depth_map is None or self._depth_source_rgb is None:
			return
		gray, color = depth_preview_images(
			self._depth_map,
			self._depth_invert_checkbox.isChecked(),
		)
		dialog = QDialog(self)
		dialog.setWindowTitle("깊이 지도 미리보기")
		dialog.resize(1200, 480)
		layout = QHBoxLayout(dialog)
		previews = (
			("원본 사진", image_from_rgb_array(self._depth_source_rgb)),
			("흑백 깊이 지도", image_from_rgb_array(gray)),
			("색상 깊이 지도", image_from_rgb_array(color)),
		)
		for title, image in previews:
			column = QVBoxLayout()
			label = QLabel(title)
			label.setAlignment(Qt.AlignmentFlag.AlignCenter)
			preview = QLabel()
			preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
			preview.setPixmap(
				QPixmap.fromImage(image).scaled(
					380,
					380,
					Qt.AspectRatioMode.KeepAspectRatio,
					Qt.TransformationMode.SmoothTransformation,
				)
			)
			column.addWidget(label)
			column.addWidget(preview)
			layout.addLayout(column)
		dialog.exec()

	def _set_media_image(self, image: QImage) -> None:
		self._current_media_image = image
		if not self._test_pattern_active and not self._stereo_mode:
			self._canvas.set_image(image)

	def _is_playback_media(self) -> bool:
		return self._is_video or self._movie is not None

	def _show_test_pattern(self) -> None:
		self._test_pattern_active = True
		self._stereo_mode = False
		self._test_pattern_button.hide()
		self._media_button.show()
		self._canvas.set_image(self._test_pattern)

	def _show_media(self) -> None:
		self._test_pattern_active = False
		self._test_pattern_button.show()
		self._media_button.hide()
		if self._stereo_mode:
			self._render_stereo_views()
		elif not self._current_media_image.isNull():
			self._canvas.set_image(self._current_media_image)

	def _choose_file(self) -> None:
		file_name, _ = QFileDialog.getOpenFileName(
			self, "미디어 선택", str(Path.home()), FILE_FILTER
		)
		if not file_name:
			return
		self._load_media(Path(file_name))

	def _load_media(self, path: Path) -> None:
		extension = path.suffix.lower()
		if extension not in SUPPORTED_EXTENSIONS:
			QMessageBox.warning(self, "지원하지 않는 형식", "지원되는 사진 또는 동영상 파일을 선택하세요.")
			return

		self._invalidate_depth_result()
		self._release_video()
		try:
			if extension in GIF_EXTENSIONS:
				supported_formats = {
					bytes(format_name).decode("ascii", errors="ignore").lower()
					for format_name in QMovie.supportedFormats()
				}
				if "gif" not in supported_formats:
					raise ValueError("현재 Qt에서 GIF 형식을 지원하지 않습니다.")
				movie = QMovie(str(path), parent=self)
				if not movie.isValid():
					movie.deleteLater()
					raise ValueError("GIF를 열 수 없거나 손상된 파일입니다.")
				movie.setCacheMode(QMovie.CacheMode.CacheNone)
				movie.frameChanged.connect(self._update_gif_frame)
				loop_setter = getattr(movie, "setLoopCount", None)
				if callable(loop_setter):
					loop_setter(-1)
				movie.finished.connect(self._restart_gif_loop)
				self._movie = movie
				self._is_video = False
				self._test_pattern_active = False
				self._test_pattern_button.show()
				self._media_button.hide()
				self._playback.show()
				self._depth_controls.hide()
				self._depth_button.setEnabled(False)
				movie.start()
			elif extension in PHOTO_EXTENSIONS:
				image = QImage(str(path))
				if image.isNull():
					raise ValueError("사진을 읽을 수 없거나 지원되지 않는 형식입니다.")
				file_stat = path.stat()
				depth_key = (str(path.resolve()).casefold(), file_stat.st_mtime_ns, file_stat.st_size)
				self._test_pattern_active = False
				self._test_pattern_button.show()
				self._media_button.hide()
				self._set_media_image(image)
				self._is_video = False
				self._playback.hide()
				self._depth_controls.show()
				self._depth_button.setEnabled(self._depth_worker is None)
				self._current_depth_key = depth_key
				cached_depth = self._depth_cache.get(depth_key)
				if cached_depth is not None:
					self._depth_map = cached_depth
					self._depth_source_rgb = qimage_to_rgb_array(image)
					self._depth_preview_button.setEnabled(True)
					self._separation_slider.setEnabled(True)
					self._focus_slider.setEnabled(True)
					self._depth_status.setText("저장된 상대 깊이 지도를 재사용할 수 있습니다.")
				else:
					self._depth_status.setText("입체 효과를 켜면 사진 깊이 분석을 시작합니다.")
			else:
				capture = cv2.VideoCapture(str(path))
				if not capture.isOpened():
					capture.release()
					raise ValueError("동영상을 열 수 없습니다. 코덱 또는 파일 상태를 확인하세요.")
				success, frame = capture.read()
				if not success or frame is None:
					capture.release()
					raise ValueError("동영상에서 첫 프레임을 읽을 수 없습니다.")
				fps = capture.get(cv2.CAP_PROP_FPS)
				if not np.isfinite(fps) or fps <= 0:
					fps = 30.0
					LOGGER.warning("유효하지 않은 FPS입니다. 30 FPS로 재생합니다: %s", path)
				self._video = capture
				self._is_video = True
				self._frame_interval_ms = max(1, round(1000 / fps))
				self._test_pattern_active = False
				self._test_pattern_button.show()
				self._media_button.hide()
				self._set_media_image(image_from_video_frame(frame))
				self._playback.show()
				self._depth_controls.hide()
				self._depth_button.setEnabled(False)
				self._timer.start(self._frame_interval_ms)
			self._pages.setCurrentIndex(1)
			LOGGER.info("미디어를 불러왔습니다: %s", path)
		except (OSError, ValueError, cv2.error) as error:
			self._release_video()
			LOGGER.exception("미디어를 불러오지 못했습니다: %s", path)
			QMessageBox.critical(self, "불러오기 실패", str(error))

	def _update_gif_frame(self, frame_number: int = -1) -> None:
		if self._movie is None:
			return
		image = self._movie.currentImage()
		if image.isNull():
			LOGGER.error("GIF의 현재 프레임을 읽지 못했습니다.")
			return
		self._set_media_image(image)

	def _restart_gif_loop(self) -> None:
		if self._movie is not None:
			self._movie.start()

	def _advance_video(self) -> None:
		if self._video is None:
			self._timer.stop()
			return
		try:
			success, frame = self._video.read()
			if not success or frame is None:
				frame_count = self._video.get(cv2.CAP_PROP_FRAME_COUNT)
				frame_position = self._video.get(cv2.CAP_PROP_POS_FRAMES)
				at_end = (
					np.isfinite(frame_count)
					and np.isfinite(frame_position)
					and frame_count > 0
					and frame_position >= frame_count - 0.5
				)
				if at_end:
					if not self._video.set(cv2.CAP_PROP_POS_FRAMES, 0):
						LOGGER.error("동영상 종료 후 처음 위치로 이동하지 못했습니다.")
						message = "동영상을 처음 위치로 이동할 수 없습니다. 파일 상태를 확인하세요."
					else:
						success, frame = self._video.read()
						if success and frame is not None:
							self._set_media_image(image_from_video_frame(frame))
							return
						LOGGER.error("동영상이 끝난 뒤 첫 프레임을 다시 읽지 못했습니다.")
						message = "동영상을 처음부터 다시 읽을 수 없습니다. 파일 상태를 확인하세요."
				else:
					LOGGER.error(
						"동영상 프레임 읽기가 종료 지점이 아닌 곳에서 실패했습니다. "
						"위치=%s, 전체 프레임=%s",
						frame_position,
						frame_count,
					)
					message = "동영상 프레임을 읽지 못했습니다. 파일 손상 여부를 확인하세요."
				self._timer.stop()
				QMessageBox.warning(self, "재생 오류", message)
				return
			self._set_media_image(image_from_video_frame(frame))
		except cv2.error:
			self._timer.stop()
			LOGGER.exception("동영상 프레임 처리 중 오류가 발생했습니다.")
			QMessageBox.critical(self, "재생 오류", "동영상 프레임을 처리하지 못했습니다.")

	def _play_media(self) -> None:
		if self._movie is not None:
			self._movie.start()
			return
		if self._video is not None:
			self._timer.start(self._frame_interval_ms)

	def _pause_media(self) -> None:
		if self._movie is not None:
			self._movie.setPaused(True)
		else:
			self._timer.stop()

	def _restart_media(self) -> None:
		if self._movie is not None:
			self._movie.stop()
			self._movie.start()
			return
		if self._video is None:
			return
		self._timer.stop()
		try:
			seek_success = self._video.set(cv2.CAP_PROP_POS_FRAMES, 0)
			success, frame = self._video.read() if seek_success else (False, None)
		except cv2.error:
			LOGGER.exception("동영상을 처음부터 다시 읽는 중 오류가 발생했습니다.")
			success, frame = False, None
		if success and frame is not None:
			self._set_media_image(image_from_video_frame(frame))
			self._timer.start(self._frame_interval_ms)
		else:
			LOGGER.error("동영상을 처음부터 다시 읽지 못했습니다.")
			QMessageBox.warning(self, "재생 오류", "동영상을 처음부터 다시 재생할 수 없습니다.")

	def _return_to_select(self) -> None:
		self._release_video()
		self._invalidate_depth_result()
		self._is_video = False
		self._current_media_image = QImage()
		self._test_pattern_active = False
		self._canvas.clear()
		self._pages.setCurrentIndex(0)

	def _toggle_full_screen(self) -> None:
		if self.isFullScreen():
			self._leave_full_screen()
			return

		self._toolbar.hide()
		self._playback.hide()
		self._correction_controls.hide()
		self._depth_controls.hide()
		self._media_layout.setContentsMargins(0, 0, 0, 0)
		self._media_layout.setSpacing(0)
		self.showFullScreen()

	def _leave_full_screen(self) -> None:
		self.showNormal()
		self._media_layout.setContentsMargins(12, 8, 12, 10)
		self._media_layout.setSpacing(6)
		self._toolbar.show()
		self._correction_controls.show()
		if not self._is_playback_media() and not self._current_media_image.isNull():
			self._depth_controls.show()
		if self._is_playback_media():
			self._playback.show()

	def keyPressEvent(self, event: QKeyEvent) -> None:
		if event.key() == Qt.Key.Key_Escape and self.isFullScreen():
			self._leave_full_screen()
			event.accept()
			return
		super().keyPressEvent(event)

	def _release_video(self) -> None:
		self._timer.stop()
		if self._video is not None:
			self._video.release()
			self._video = None
		if self._movie is not None:
			self._movie.frameChanged.disconnect(self._update_gif_frame)
			self._movie.finished.disconnect(self._restart_gif_loop)
			self._movie.stop()
			self._movie.setFileName("")
			self._movie.deleteLater()
			self._movie = None

	def closeEvent(self, event) -> None:
		self._release_video()
		if self._depth_worker is not None and self._depth_worker.isRunning():
			self._depth_worker.cancel()
			self._depth_worker.wait()
		event.accept()


def main() -> int:
	app = QApplication(sys.argv)
	window = QuadViewWindow()
	window.show()
	return app.exec()


if __name__ == "__main__":
	sys.exit(main())
