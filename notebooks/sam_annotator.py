import sys
import os
import json
from copy import deepcopy
from PyQt5.QtCore import Qt, QPoint, QEvent, QRectF, QObject, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPainter, QPen, QPixmap, QIcon
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
        self.labels = np.array([])
        self.cls_labels = ["CT", "Terroist", "Corpse"]
        self.cls_num = 1

    def predict(self, pos_x, pos_y):
        if self.pos_prompts.size == 0:
            input_points = np.array([[pos_x, pos_y]])
            input_labels = np.array([1])
        else:
            input_points = np.concatenate(
                (self.pos_prompts, np.array([[pos_x, pos_y]])), axis=0
            )
            input_labels = np.concatenate((self.labels, np.array([1])), axis=0)

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
        self.setWindowIcon(QIcon("./as.jpg"))
        self.sam = SAM()
        self.my_signal.connect(self.sam.predict)
        self.initUI()
        self.image_paths = []  # 添加一个变量来存储所有图片的路径
        self.mask_labels = {}

    def open_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Open Folder")
        if folder_path:
            self.image_paths = []
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".png") or file_name.endswith(".jpg"):
                    self.image_paths.append(os.path.join(folder_path, file_name))
            if self.image_paths:
                self.current_image_index = 0
                print("image_path: ", self.image_paths[self.current_image_index])
                self.open_image(self.image_paths[self.current_image_index])

    def keyPressEvent(self, event):
        if event.text().isdigit():
            cls_num = int(event.text())
            if cls_num >= 0:
                self.sam.cls_num = cls_num
                if cls_num < len(self.sam.cls_labels):
                    self.text_edit.append(
                        f"Current label is {self.sam.cls_labels[cls_num]}."
                    )
                else:
                    self.text_edit.append(
                        f"{cls_num} is an invaild class label number."
                    )
        elif event.text() == "j":
            if len(self.image_paths) > 0:
                self.current_image_index = (self.current_image_index - 1) % len(
                    self.image_paths
                )
                self.open_image(self.image_paths[self.current_image_index])
            else:
                self.text_edit.append("Please open a folder.")
        elif event.text() == "k":
            if len(self.image_paths) > 0:
                self.current_image_index = (self.current_image_index + 1) % len(
                    self.image_paths
                )
                self.open_image(self.image_paths[self.current_image_index])
            else:
                self.text_edit.append("Please open a folder.")
        else:
            super().keyPressEvent(event)

    def initUI(self):
        self.setWindowTitle("SAM Annotator")

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

        open_button = QPushButton("Open Image [Ctrl+O]", self)  # 创建打开图像按钮
        open_button.setShortcut("Ctrl+O")  # 添加快捷键
        open_button.clicked.connect(self.open_image)  # 绑定打开图像函数
        layout.addWidget(open_button)  # 添加按钮到布局中

        clear_button = QPushButton("Clear Prompts [Backspace]", self)  # 创建清空五角星按钮
        clear_button.setShortcut("Backspace")
        clear_button.clicked.connect(self.clear_stars)  # 绑定清空五角星函数
        layout.addWidget(clear_button)  # 添加按钮到布局中

        open_folder_button = QPushButton("Open Folder [Ctrl+K]", self)
        open_folder_button.setShortcut("Ctrl+K")
        open_folder_button.clicked.connect(self.open_folder)
        layout.addWidget(open_folder_button)

        self.image = QImage()  # 创建图像对象
        self.image_path = None  # 初始化图像路径为空

        self.show()  # 显示窗口
        save_as_button = QPushButton("Save Mask As [Ctrl+Alt+S]", self)  # 创建保存mask按钮
        save_as_button.setShortcut("Ctrl+Alt+S")
        save_as_button.clicked.connect(self.save_as_mask)  # 绑定保存mask函数
        layout.addWidget(save_as_button)  # 添加按钮到布局中

        save_button = QPushButton("Save Mask [Ctrl+S]", self)  # 创建保存mask按钮
        save_button.setShortcut("Ctrl+S")
        save_button.clicked.connect(self.save_mask)  # 绑定保存mask函数
        layout.addWidget(save_button)  # 添加按钮到布局中

    def save_mask(self):
        if self.sam.mask is not None:
            if self.image_path is not None:
                base_file_name = (
                    os.path.splitext(os.path.basename(self.image_path))[0] + "_mask.npz"
                )
                first_file_name = os.path.join(
                    os.path.dirname(self.image_path),
                    os.path.splitext(os.path.basename(base_file_name))[0]
                    + f"_{1}"
                    + os.path.splitext(os.path.basename(base_file_name))[1],
                )
                file_name = os.path.join(
                    os.path.dirname(self.image_path), first_file_name
                )
                # 生成最小外接矩形
                contours, _ = cv2.findContours(
                    self.sam.mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )

                contour_sizes = [
                    (cv2.contourArea(contour), contour) for contour in contours
                ]
                largest_contour = max(contour_sizes, key=lambda x: x[0])[1]
                rect = cv2.minAreaRect(largest_contour)

                if os.path.exists(file_name):
                    i = 1
                    while True:
                        new_file_name = os.path.join(
                            os.path.dirname(self.image_path),
                            os.path.splitext(os.path.basename(base_file_name))[0]
                            + f"_{i}"
                            + os.path.splitext(os.path.basename(base_file_name))[1],
                        )
                        if not os.path.exists(new_file_name):
                            file_name = new_file_name
                            break
                        i += 1
                np.savez_compressed(file_name, mask_rle=self.sam.mask.astype(np.uint8))
                self.mask_labels[os.path.basename(file_name)] = {
                    "class_num": self.sam.cls_num,
                    "class_name": self.sam.cls_labels[self.sam.cls_num],
                    "bbox": [rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]],
                }
                folder_path = os.path.dirname(file_name)
                mask_labels_path = os.path.join(folder_path, "mask_labels.json")
                with open(mask_labels_path, "w") as f:
                    json.dump(self.mask_labels, f)

                self.text_edit.append(f"Mask saved as {file_name}")
                self.clear_stars()
            else:
                self.text_edit.append("Please open an image first.")
        else:
            self.text_edit.append("No mask to save.")

    def save_as_mask(self):
        if self.sam.mask is not None:
            default_file_name = (
                os.path.splitext(os.path.basename(self.image_path))[0] + "_mask.npz"
            )
            file_name, _ = QFileDialog.getSaveFileName(
                self,
                "Save Mask",
                default_file_name,
                "Numpy Files (*.npz *.npy);;All Files (*)",
            )
            if file_name:
                # 存储成npz格式文件
                np.savez_compressed(file_name, mask_rle=self.sam.mask.astype(np.uint8))
                self.mask_labels[os.path.basename(file_name)] = self.sam.cls_num
                folder_path = os.path.dirname(file_name)
                mask_labels_path = os.path.join(folder_path, "mask_labels.json")
                with open(mask_labels_path, "w") as f:
                    json.dump(self.mask_labels, f)

    def decompress(self, file_name):
        data = np.load(file_name)
        mask_rle = data["mask_rle"]
        data.close()
        return mask_rle

    def rle_encode(self, mask):
        pixels = mask.T.flatten()
        count = 0
        for i in range(len(pixels) - 1):
            if pixels[i] == 0:
                count += 1
            else:
                break
        encoded_msg = [count]
        i = count
        while i < len(pixels) - 1:
            count = 1
            j = i
            while j < len(pixels) - 1:
                if pixels[j] == pixels[j + 1]:
                    count += 1
                    j += 1
                else:
                    break
            i = j + 1
            encoded_msg.append(count)
        return np.array(encoded_msg)

    def decode(self, rle):
        if rle[0] != 0:
            decoded_msg = [0] * rle[0]
        else:
            decoded_msg = []

        for i in range(1, len(rle)):
            if i % 2 == 0:
                decoded_msg.extend([0] * rle[i])
            else:
                decoded_msg.extend([1] * rle[i])
        return decoded_msg

    def clear_stars(self):
        self.image = self.orgin_img.copy()
        self.sam.pos_prompts = np.array([[]])
        self.sam.labels = np.array([])
        self.resize_image()

    def predict(self, pos_x, pos_y):
        self.sam.predict(pos_x, pos_y)
        mask_image = self.sam.mask
        if not self.image.isNull():
            h, w = mask_image.shape[-2:]
            np_img = self.qimage_to_numpy(self.image)
            color = np.random.randint(0, 256, size=4)
            mask_x, mask_y = np.where(mask_image)
            np_img[mask_x, mask_y, :] = 0.4 * color + 0.6 * np_img[mask_x, mask_y, :]
            pixmap = self.numpy_to_qpixmap(np_img)
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

    def open_image(self, file_name=None):
        if not file_name:
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
            folder_path = os.path.dirname(file_name)
            mask_labels_path = os.path.join(folder_path, "mask_labels.json")
            if os.path.exists(mask_labels_path):
                with open(mask_labels_path, "r") as f:
                    self.mask_labels = json.load(f)
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
                    img_pos = self.graphics_view.mapToScene(event.pos()).toPoint()
                    img_pos.setX(int((img_pos.x() - offset.x()) / scale_ratio_w))
                    img_pos.setY(int((img_pos.y() - offset.y()) / scale_ratio_h))
                    if self.sam.pos_prompts.size == 0:
                        self.sam.pos_prompts = np.array([[img_pos.x(), img_pos.y()]])
                        if event.button() == Qt.LeftButton:
                            self.sam.labels = np.array([1])
                        elif event.button() == Qt.RightButton:
                            self.sam.labels = np.array([-1])

                    else:
                        self.sam.pos_prompts = np.concatenate(
                            (
                                self.sam.pos_prompts,
                                np.array([[img_pos.x(), img_pos.y()]]),
                            ),
                            axis=0,
                        )
                        if event.button() == Qt.LeftButton:
                            self.sam.labels = np.concatenate(
                                (self.sam.labels, np.array([1])), axis=0
                            )
                        elif event.button() == Qt.RightButton:
                            self.sam.labels = np.concatenate(
                                (self.sam.labels, np.array([-1])), axis=0
                            )
                    painter = QPainter(pixmap)
                    if event.button() == Qt.LeftButton:
                        painter.setPen(QPen(Qt.red, 3, Qt.SolidLine))
                    elif event.button() == Qt.RightButton:
                        painter.setPen(QPen(Qt.blue, 3, Qt.SolidLine))

                    painter.drawPolygon(
                        img_pos,
                        img_pos + QPoint(10, 0),
                        img_pos + QPoint(5, 10),
                        img_pos - QPoint(5, 10),
                        img_pos - QPoint(10, 0),
                    )
                    painter.end()
                    self.image = pixmap.toImage()
                    self.predict(img_pos.x(), img_pos.y())

        return super().eventFilter(source, event)

    def numpy_to_qimage(self, image):
        height, width, channels = image.shape
        bytes_per_line = channels * width
        qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format_ARGB32)
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
