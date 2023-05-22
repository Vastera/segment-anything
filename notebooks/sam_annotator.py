import sys
import os
from copy import deepcopy
from PyQt5.QtCore import Qt, QPoint, QEvent, QRectF, QObject, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QGraphicsScene,
    QGraphicsView,
    QLabel,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor


class SAM(QObject):
    def __init__(self):
        sam_checkpoint = "../sam_vit_b_01ec64.pth"
        model_type = "vit_b"
        device = "cuda"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        self.predictor = SamPredictor(sam)
        self.mask = None
        self.pos_prompts = np.array([[]])
        self.neg_prompts = np.array([[]])
        self.labels = np.array([])

    def predict(self, pos_x, pos_y):
        input_points = np.concatenate((self.pos_prompts, np.array([[pos_x, pos_y]])), axis=0)
        input_labels = np.concatenate((self.labels, np.array([1])), axis=1)
        masks, scores, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
        )
        self.mask = masks[np.argmax(scores)]

    def open_image(self, image_path):
        self.cv_img = cv2.imread(image_path)
        self.cv_img = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(self.cv_img)


class ImageApp(QMainWindow):
    my_signal = pyqtSignal(int, int)

    def __init__(self):
        super().__init__()

        self.sam = SAM()
        self.my_signal.connect(self.sam.predict)
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Image Viewer")

        central_widget = QWidget()  # 创建中央窗口
        self.setCentralWidget(central_widget)  # 设置中央窗口

        layout = QVBoxLayout(central_widget)  # 创建垂直布局

        self.graphics_view = QGraphicsView(self)  # 创建图形视图
        self.graphics_view.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff
        )  # 设置水平滚动条不可见
        self.graphics_view.setVerticalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff
        )  # 设置垂直滚动条不可见
        self.graphics_view.setAlignment(Qt.AlignCenter)  # 设置图像居中
        layout.addWidget(self.graphics_view, stretch=5)  # 添加图形视图到布局中，高度比例为5:1

        self.text_edit = QTextEdit(self)  # 创建文本框
        self.text_edit.setMinimumHeight(50)  # 设置最小高度
        self.text_edit.setMaximumHeight(150)  # 设置最大高度
        self.text_edit.setReadOnly(True)  # 设置只读
        layout.addWidget(self.text_edit, stretch=1)  # 添加文本框到布局中，高度比例为5:1

        open_button = QPushButton("Open Image", self)  # 创建打开图像按钮
        open_button.clicked.connect(self.open_image)  # 绑定打开图像函数
        layout.addWidget(open_button)  # 添加按钮到布局中

        clear_button = QPushButton("Clear Prompts", self)  # 创建清空五角星按钮
        clear_button.clicked.connect(self.clear_stars)  # 绑定清空五角星函数
        layout.addWidget(clear_button)  # 添加按钮到布局中

        self.image = QImage()  # 创建图像对象
        self.image_path = None  # 初始化图像路径为空

        self.show()  # 显示窗口

    def clear_stars(self):
        self.image = self.orgin_img.copy()
        self.sam.pos_prompts = np.array([[]])
        self.sam.neg_prompts = np.array([[]])
        self.sam.labels = np.array([])
        self.resize_image()

    def predict(self, pos_x, pos_y):
        self.sam.predict(pos_x, pos_y)
        mask_image = self.sam.mask
        if not self.image.isNull():
            h, w = mask_image.shape[-2:]
            # color = np.concatenate([np.random.random(3) * 255, np.array([0.1])], axis=0)
            color = np.random.random(3) * 255
            mask_image = mask_image.reshape(h, w, 1) * color.reshape(1, 1, -1).astype(
                np.uint8
            )
            np_img = self.qimage_to_numpy(self.image)
            combined_img = cv2.addWeighted(mask_image, 0.4, np_img[:,:,:3], 0.6, 0)
            pixmap = self.numpy_to_qpixmap(combined_img)
            tmp_image = pixmap.toImage()
            self.graphics_view.setRenderHint(QPainter.Antialiasing)
            self.graphics_view.setRenderHint(QPainter.SmoothPixmapTransform)
            self.graphics_view.setMouseTracking(True)
            self.graphics_view.viewport().installEventFilter(self)
            self.resize_image(tmp_image)

    def resize_image(self, new_image=None):
        if not self.image.isNull():
            pixmap = QPixmap.fromImage(self.image if not new_image else new_image)
            scaled_pixmap = pixmap.scaled(
                self.graphics_view.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation
            )
            scene = QGraphicsScene(self)
            scene.addPixmap(scaled_pixmap)
            self.graphics_view.setScene(scene)
            self.graphics_view.setSceneRect(QRectF(scaled_pixmap.rect()))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resize_image()

    def open_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Images (*.png *.xpm *.jpg *.bmp);;All Files (*)",
            options=options,
        )
        if file_name:
            self.image_path = file_name
            self.image.load(file_name)
            self.orgin_img = self.image.copy()
            self.sam.open_image(file_name)
            self.graphics_view.setRenderHint(QPainter.Antialiasing)
            self.graphics_view.setRenderHint(QPainter.SmoothPixmapTransform)
            self.graphics_view.setMouseTracking(True)
            self.graphics_view.viewport().installEventFilter(self)
            self.resize_image()

    def eventFilter(self, source, event):
        if source == self.graphics_view.viewport():
            if event.type() == QEvent.MouseMove:  # QEvent.Type.MouseMove
                pos = event.pos()
                offset = self.graphics_view.mapToScene(QPoint(0, 0))
                scale_ratio_w = (
                    self.graphics_view.sceneRect().width() / self.image.width()
                )
                scale_ratio_h = (
                    self.graphics_view.sceneRect().height() / self.image.height()
                )
                img_pos = self.graphics_view.mapToScene(pos).toPoint()
                img_pos.setX(int((img_pos.x() - offset.x()) / scale_ratio_w))
                img_pos.setY(int((img_pos.y() - offset.y()) / scale_ratio_h))

                prompt_pos = np.array([img_pos.x(), img_pos.y()])

                self.text_edit.setText(
                    f"Mouse position: ({prompt_pos[0]}, {prompt_pos[1]})"
                )
                self.predict(prompt_pos[0], prompt_pos[1])
            elif (
                event.type() == QEvent.MouseButtonPress
            ):  # QEvent.Type.MouseButtonPress
                if self.image_path:
                    # 获取缩放比例
                    pixmap = QPixmap.fromImage(self.image)
                    offset = self.graphics_view.mapToScene(QPoint(0, 0))
                    scale_ratio_w = (
                        self.graphics_view.sceneRect().width() / self.image.width()
                    )
                    scale_ratio_h = (
                        self.graphics_view.sceneRect().height() / self.image.height()
                    )

                    # 将鼠标点击的位置映射回原始图像的坐标
                    # 获取缩放比例
                    img_pos = self.graphics_view.mapToScene(event.pos()).toPoint()
                    img_pos.setX(int((img_pos.x() - offset.x()) / scale_ratio_w))
                    img_pos.setY(int((img_pos.y() - offset.y()) / scale_ratio_h))
                    self.sam.pos_prompts = np.concatenate((self.sam.pos_prompts, np.array([[pos_x, pos_y]])), axis=0)
                    self.sam.labels = np.concatenate((self.sam.labels, np.array([1])), axis=1)

                    painter = QPainter(pixmap)
                    painter.setPen(QPen(Qt.red, 3, Qt.SolidLine))
                    painter.drawPolygon(
                        img_pos,
                        img_pos + QPoint(10, 0),
                        img_pos + QPoint(5, 10),
                        img_pos - QPoint(5, 10),
                        img_pos - QPoint(10, 0),
                    )
                    painter.end()
                    self.image = pixmap.toImage()
                    self.resize_image()
        return super().eventFilter(source, event)

    def numpy_to_qimage(self, image):
        height, width, channels = image.shape
        bytes_per_line = channels * width
        qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return qimage

    def qimage_to_numpy(self, qimage):
        width = qimage.width()
        height = qimage.height()
        bytes_per_line = qimage.bytesPerLine()
        image_format = qimage.format()

        if image_format == QImage.Format_RGB888:
            return np.array(
                qimage.bits().asarray(height * bytes_per_line), dtype=np.uint8
            ).reshape(height, width, 3)
        elif image_format == QImage.Format_Grayscale8:
            return np.array(
                qimage.bits().asarray(height * bytes_per_line), dtype=np.uint8
            ).reshape(height, width)
        elif image_format in [QImage.Format_ARGB32, 4]:
            return np.array(
                qimage.bits().asarray(height * bytes_per_line), dtype=np.uint8
            ).reshape(height, width, 4)
        else:
            raise ValueError(f"Unsupported image format: {image_format}")


    def numpy_to_qpixmap(self, image):
        qimage = self.numpy_to_qimage(image)
        pixmap = QPixmap.fromImage(qimage)
        return pixmap


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = ImageApp()
    sys.exit(app.exec_())