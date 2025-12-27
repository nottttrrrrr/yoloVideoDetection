import sys
import cv2
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtGui import QFont, QColor, QPalette
from PyQt6.QtWidgets import QGraphicsDropShadowEffect
from ultralytics import YOLO

# --- 自定义样式表 (QSS) ---
# 这里定义了全局的颜色、圆角、按钮样式等
STYLESHEET = """
QMainWindow {
    background-color: #1e1e2e;
}
QWidget {
    font-family: "Segoe UI", "Microsoft YaHei";
    font-size: 14px;
    color: #cdd6f4;
}
QGroupBox {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 12px;
    margin-top: 10px;
    font-weight: bold;
    color: #89b4fa;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 15px;
    padding: 0 5px;
    background-color: #1e1e2e; /* 标题背景与主背景融合 */
}
QLabel#TitleLabel {
    font-size: 24px;
    font-weight: bold;
    color: #89b4fa;
    padding: 10px;
}
QLabel#VideoLabel {
    background-color: #11111b;
    border: 2px dashed #45475a;
    border-radius: 8px;
    color: #6c7086;
}
QTextBrowser {
    background-color: #181825;
    border: 1px solid #313244;
    border-radius: 8px;
    padding: 10px;
    color: #a6adc8;
    font-family: "Consolas", monospace;
    font-size: 13px;
}
QPushButton {
    background-color: #89b4fa;
    color: #1e1e2e;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: bold;
    min-height: 25px;
}
QPushButton:hover {
    background-color: #b4befe;
}
QPushButton:pressed {
    background-color: #74c7ec;
}
QPushButton:disabled {
    background-color: #45475a;
    color: #6c7086;
}
QPushButton#StopButton {
    background-color: #f38ba8; /* 红色停止按钮 */
    color: #1e1e2e;
}
QPushButton#StopButton:hover {
    background-color: #fab387;
}
QCheckBox {
    spacing: 8px;
    color: #cdd6f4;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 2px solid #585b70;
}
QCheckBox::indicator:checked {
    background-color: #a6e3a1; /* 绿色选中 */
    border-color: #a6e3a1;
}
"""


class ImagePopup(QtWidgets.QDialog):
    def __init__(self, image, parent=None):
        super().__init__(parent)
        self.image = image
        self.initUI()

    def initUI(self):
        self.setWindowTitle("🔍 详情查看")
        self.resize(1200, 800)
        self.setStyleSheet("background-color: #1e1e2e;")

        layout = QtWidgets.QVBoxLayout(self)
        self.label = QtWidgets.QLabel(self)
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        pixmap = QtGui.QPixmap.fromImage(self.image)
        scaled_pixmap = pixmap.scaled(1180, 780,
                                      QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                      QtCore.Qt.TransformationMode.SmoothTransformation)
        self.label.setPixmap(scaled_pixmap)
        layout.addWidget(self.label)


class VideoThread(QtCore.QThread):
    updateFrame = QtCore.pyqtSignal(QtGui.QImage)
    results = QtCore.pyqtSignal(list)

    def __init__(self, video_file, model, classIndexes, parent=None):
        super().__init__(parent)
        self.video_file = video_file
        self.model = model
        self.running = False
        self.paused = False  # 控制是否暂停处理
        self.classIndexes = classIndexes

    def run(self):
        self.running = True
        self.paused = False  # 控制是否暂停处理
        cap = cv2.VideoCapture(self.video_file)
        while self.running and cap.isOpened():
            # --- 关键修改：暂停逻辑 ---
            if self.paused:
                self.msleep(100)  # 线程休眠100ms，降低CPU占用，等待唤醒
                continue  # 跳过本次循环，不读取下一帧
            # ------------------------
            ret, frame = cap.read()
            if not ret:
                break

            # 使用模型检测
            try:
                results = self.model(frame, stream=True, classes=self.classIndexes)
                results_list = list(results)

                # 绘制结果
                if results_list:
                    annotated_img = results_list[0].plot()
                    # 转换颜色空间 BGR -> RGB
                    rgb_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_img.shape
                    bytes_per_line = ch * w
                    qimg = QtGui.QImage(rgb_img.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)

                    self.updateFrame.emit(qimg)
                    self.results.emit(results_list)
            except Exception as e:
                print(f"Error in thread: {e}")
                break
            # 简单的帧率控制，防止界面卡顿
            self.msleep(30)

        cap.release()
        self.running = False

    def stop(self):
        # self.running = False
        """彻底结束线程（切换视频或关闭软件时用）"""
        self.running = False
        self.wait()  # 等待线程安全退出

    def pause_video(self):
        """暂停播放（不释放资源，记住进度）"""
        self.paused = True

    def continue_video(self):
        """恢复播放"""
        self.paused = False
    def setClassIndexes(self, classIndexes):
        self.classIndexes = classIndexes


class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.videoThread = None
        # self.model = YOLO("yolov8n.pt")  # 确保你有模型文件
        self.model = YOLO("models/best_last.pt")
        self.detected_image = None

        self.current_frame = None  # <--- 新增：用于存储当前待放大的画面


        self.setupUi()
        self.apply_stylesheet()

        # 初始化状态
        self.zoomButton.setDisabled(True)
        self.stopButton.setDisabled(True)
        self.continueButton.setDisabled(True)

        # 默认全部选中
        for cb in self.checkboxes:
            cb.setChecked(True)

    def apply_stylesheet(self):
        self.setStyleSheet(STYLESHEET)

    def setupUi(self):
        self.setObjectName("MainWindow")
        self.resize(1200, 800)
        self.setWindowTitle("课堂行为智能监测系统")

        # 主窗口部件
        self.centralwidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralwidget)

        # 全局布局 (垂直)
        main_layout = QtWidgets.QVBoxLayout(self.centralwidget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)

        # --- 1. 顶部标题栏 ---
        header_layout = QtWidgets.QHBoxLayout()
        title_label = QtWidgets.QLabel("🎓 课堂行为智能监测分析系统")
        title_label.setObjectName("TitleLabel")
        title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(title_label)
        main_layout.addLayout(header_layout)

        # --- 2. 中间内容区 (左侧视频 + 右侧统计) ---
        content_layout = QtWidgets.QHBoxLayout()

        # 左侧：视频显示区
        video_group = QtWidgets.QGroupBox("")
        video_layout = QtWidgets.QVBoxLayout(video_group)
        video_layout.setContentsMargins(10, 20, 10, 10)

        self.videoLabel = QtWidgets.QLabel("暂无视频源\n请上传图片或视频开始检测")
        self.videoLabel.setObjectName("VideoLabel")
        self.videoLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.videoLabel.setMinimumSize(640, 360)
        self.videoLabel.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        video_layout.addWidget(self.videoLabel)

        content_layout.addWidget(video_group, stretch=2)  # 视频占 2/3 宽度

        # 右侧：统计分析区
        stats_group = QtWidgets.QGroupBox("📊 实时分析数据")
        stats_layout = QtWidgets.QVBoxLayout(stats_group)
        stats_layout.setContentsMargins(15, 25, 15, 15)

        self.textBrowser = QtWidgets.QTextBrowser()
        self.textBrowser.setHtml(
            "<div style='text-align:center; color:#6c7086; margin-top:50px;'>等待分析数据...</div>")
        stats_layout.addWidget(self.textBrowser)

        content_layout.addWidget(stats_group, stretch=1)  # 统计占 1/3 宽度

        main_layout.addLayout(content_layout, stretch=1)

        # --- 3. 底部控制面板 ---
        control_group = QtWidgets.QGroupBox("🛠️ 控制中心")
        control_group.setFixedHeight(160)
        control_layout = QtWidgets.QVBoxLayout(control_group)
        control_layout.setContentsMargins(20, 30, 20, 20)

        # 3.1 行为复选框行
        check_layout = QtWidgets.QHBoxLayout()
        check_layout.addWidget(QtWidgets.QLabel("监测目标："))

        self.checkboxes = []
        labels = ["✋ 举手", "📖 看书", "✍️ 写字", "📱 玩手机", "🙇 低头", "😴 睡觉"]
        for label_text in labels:
            cb = QtWidgets.QCheckBox(label_text)
            cb.stateChanged.connect(self.updateCheckBoxState)
            self.checkboxes.append(cb)
            check_layout.addWidget(cb)

        check_layout.addStretch()  # 弹簧，把复选框顶到左边
        control_layout.addLayout(check_layout)

        # 分割线
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        line.setStyleSheet("background-color: #45475a;")
        control_layout.addWidget(line)

        # 3.2 按钮操作行
        btn_layout = QtWidgets.QHBoxLayout()

        self.btn_img = QtWidgets.QPushButton("🖼️ 图片监测")
        self.btn_video = QtWidgets.QPushButton("🎥 视频监测")
        self.zoomButton = QtWidgets.QPushButton("🔍 放大查看")
        self.continueButton = QtWidgets.QPushButton("▶️ 继续")
        self.stopButton = QtWidgets.QPushButton("⏹️ 停止")
        self.stopButton.setObjectName("StopButton")  # 设置ID以应用红色样式

        # 按钮样式微调（更宽一点）
        for btn in [self.btn_img, self.btn_video, self.zoomButton, self.continueButton, self.stopButton]:
            btn.setMinimumWidth(100)
            btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

        btn_layout.addWidget(self.btn_img)
        btn_layout.addWidget(self.btn_video)
        btn_layout.addStretch()  # 中间弹簧
        btn_layout.addWidget(self.zoomButton)
        btn_layout.addWidget(self.continueButton)
        btn_layout.addWidget(self.stopButton)

        control_layout.addLayout(btn_layout)
        main_layout.addWidget(control_group)

        # 连接信号
        self.btn_img.clicked.connect(self.openImageFile)
        self.btn_video.clicked.connect(self.openVideoFile)
        self.zoomButton.clicked.connect(self.showZoomedImage)
        self.stopButton.clicked.connect(self.stopMonitoring)
        self.continueButton.clicked.connect(self.continueMonitoring)

        # 初始加载封面
        self.setCoverImage()

    # --- 逻辑功能 (保持原有逻辑框架，稍作优化) ---

    def setCoverImage(self):
        # 我们可以用纯色或者占位符替代封面，防止找不到图片报错
        # 你可以把原来的 'cover3.jpg' 放回来
        pass

    def SelectClass(self):
        return [i for i, cb in enumerate(self.checkboxes) if cb.isChecked()]

    def updateCheckBoxState(self):
        if self.videoThread is not None:
            self.videoThread.setClassIndexes(self.SelectClass())

    def openImageFile(self):
        if not any(cb.isChecked() for cb in self.checkboxes):
            QtWidgets.QMessageBox.warning(self, "提示", "请至少勾选一种监测行为！")
            return

        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择图片", "", "Images (*.png *.jpg *.jpeg)")  # 修正了Native Dialog问题

        if file_name:
            img = cv2.imread(file_name)
            if img is None: return

            class_idx = self.SelectClass()
            results = self.model(img, classes=class_idx)  # 同步推理

            # 处理结果
            res = list(results)[0]
            annotated_img = res.plot()
            self.detected_image = annotated_img  # 保存给放大功能用

            # 显示图片
            rgb_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_img.shape
            qimg = QtGui.QImage(rgb_img.data, w, h, ch * w, QtGui.QImage.Format.Format_RGB888)

            # 存入变量
            self.current_frame = qimg.copy()  # 使用 copy 确保数据独立

            pixmap = QtGui.QPixmap.fromImage(qimg)
            self.videoLabel.setPixmap(pixmap.scaled(
                self.videoLabel.size(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation
            ))

            # 更新文本
            self.AnalyzeResults([res])

            # 更新按钮状态
            self.zoomButton.setDisabled(False)
            self.continueButton.setDisabled(True)
            self.stopButton.setDisabled(True)

    def openVideoFile(self):
        if not any(cb.isChecked() for cb in self.checkboxes):
            QtWidgets.QMessageBox.warning(self, "提示", "请至少勾选一种监测行为！")
            return

        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择视频", "", "Videos (*.mp4 *.avi)")

        if file_name:
            # if self.videoThread is not None:
            #     self.videoThread.stop()
            #     self.videoThread.wait()
            # --- 关键：彻底清理旧线程 ---
            if self.videoThread is not None:
                self.videoThread.stop()  # 让循环结束
                self.videoThread.deleteLater()  # 标记垃圾回收
                self.videoThread = None  # 显式置空
            # -------------------------

            # 创建新线程
            self.videoThread = VideoThread(file_name, self.model, self.SelectClass(), self)
            self.videoThread.updateFrame.connect(self.updateVideoFrame)
            self.videoThread.results.connect(self.AnalyzeResults)
            self.videoThread.start()

            self.stopButton.setDisabled(False)
            self.continueButton.setDisabled(True)
            self.zoomButton.setDisabled(True)
            self.btn_img.setDisabled(True)
            self.btn_video.setDisabled(True)

    # def stopMonitoring(self):
    #     if self.videoThread:
    #         self.videoThread.stop()
    #     self.stopButton.setDisabled(True)
    #     self.continueButton.setDisabled(False)
    #     self.btn_img.setDisabled(False)
    #     self.btn_video.setDisabled(False)
        # 找到这两个函数进行替换
    def stopMonitoring(self):
        """点击暂停/停止监测"""
        if self.videoThread is not None and self.videoThread.isRunning():
            self.videoThread.pause_video()  # 只暂停，不销毁

            # 更新UI状态
            self.stopButton.setDisabled(True)
            self.continueButton.setDisabled(False)

            # 允许此时操作其他按钮（可选）
            self.btn_img.setDisabled(False)
            self.btn_video.setDisabled(False)
            self.zoomButton.setDisabled(False)  # 暂停时允许放大查看当前帧
    # def continueMonitoring(self):
    #     if self.videoThread:
    #         self.videoThread.continue_video()  # 这里需要注意Thread逻辑，简单起见重新start或resume
    #         # 由于Thread模型比较简单，这里建议直接在run里改用暂停标志位，或者重新运行
    #         self.videoThread.start()
    #
    #     self.continueButton.setDisabled(True)
    #     self.stopButton.setDisabled(False)
    #     self.btn_img.setDisabled(True)
    #     self.btn_video.setDisabled(True)
    def continueMonitoring(self):
        """点击继续监测"""
        if self.videoThread is not None and self.videoThread.isRunning():
            self.videoThread.continue_video()  # 恢复标志位

            # 更新UI状态
            self.continueButton.setDisabled(True)
            self.stopButton.setDisabled(False)

            # 继续播放时禁用其他干扰按钮
            self.btn_img.setDisabled(True)
            self.btn_video.setDisabled(True)
            self.zoomButton.setDisabled(True)
    def showZoomedImage(self):
        # if self.detected_image is not None:
        #     rgb_img = cv2.cvtColor(self.detected_image, cv2.COLOR_BGR2RGB)
        #     h, w, ch = rgb_img.shape
        #     qimg = QtGui.QImage(rgb_img.data, w, h, ch * w, QtGui.QImage.Format.Format_RGB888)
        #     popup = ImagePopup(qimg, self)
        #     popup.exec()
        if self.current_frame is not None:
            # 直接把保存的 QImage 传给弹窗类
            popup = ImagePopup(self.current_frame, self)
            popup.exec()
        else:
            QtWidgets.QMessageBox.information(self, "提示", "当前没有可放大的画面")

    def updateVideoFrame(self, qimg):
        # 1. 保存当前画面给“放大查看”按钮用
        self.current_frame = qimg

        pixmap = QtGui.QPixmap.fromImage(qimg)
        self.videoLabel.setPixmap(pixmap.scaled(
            self.videoLabel.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation
        ))

    def AnalyzeResults(self, results):
        # 统计逻辑
        class_counts = {}
        total_conf = 0.0
        count_all = 0

        classnames = ['举手', '看书', '写字', '玩手机', '低头', '睡觉']

        # 构造 HTML 格式的统计文本
        html_content = """
        <h3 style="color:#89b4fa; margin-bottom:10px;">📊 实时统计报告</h3>
        <table style="width:100%; font-size:14px; color:#cdd6f4;">
        <tr><th align="left">行为类别</th><th align="center">人数</th><th align="right">平均置信度</th></tr>
        <tr><td colspan="3"><hr style="border:1px solid #45475a;"></td></tr>
        """

        temp_data = {}  # 用于暂存每个类别的总分和数量

        for r in results:
            for box in r.boxes:
                cls = int(box.cls.item())
                conf = float(box.conf.item())

                class_counts[cls] = class_counts.get(cls, 0) + 1

                if cls not in temp_data: temp_data[cls] = []
                temp_data[cls].append(conf)

        if not class_counts:
            html_content += "<tr><td colspan='3' align='center' style='padding:20px; color:#6c7086;'>暂无检测目标</td></tr>"
        else:
            for cls_idx, count in class_counts.items():
                if cls_idx < len(classnames):
                    name = classnames[cls_idx]
                    avg = sum(temp_data[cls_idx]) / len(temp_data[cls_idx])
                    html_content += f"""
                    <tr>
                        <td style="padding:5px;">{name}</td>
                        <td align="center" style="color:#a6e3a1; font-weight:bold;">{count}</td>
                        <td align="right" style="color:#fab387;">{avg:.2f}</td>
                    </tr>
                    """

            total_people = sum(class_counts.values())
            html_content += f"""
            <tr><td colspan="3"><hr style="border:1px solid #45475a;"></td></tr>
            <tr>
                <td style="font-weight:bold;">总计人数</td>
                <td align="center" style="font-size:16px; color:#f9e2af; font-weight:bold;">{total_people}</td>
                <td></td>
            </tr>
            """

        html_content += "</table>"
        self.textBrowser.setHtml(html_content)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Ui_MainWindow()
    window.show()
    sys.exit(app.exec())