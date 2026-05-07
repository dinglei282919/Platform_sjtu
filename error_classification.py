# error_classification.py
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib

matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


def seed_torch(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# 可集中修改的实验设定值（在此区域修改超参与噪声相关设定，便于实验复现与调参）
# 注释放在同行右侧，减少占用行数
# =========================
DEFAULT_SEED = 42                       # 随机种子（用于保证结果可复现）
DEFAULT_EPOCHS = 50                     # 默认训练轮次（作为 UI 的默认值，也可在运行时覆盖）
DEFAULT_BATCH_SIZE = 32                 # 默认批次大小（UI 默认值）
DEFAULT_LR = 0.001                      # 默认学习率（UI 默认值）

NUM_TRAIN_SAMPLES = 500                 # 合成数据训练样本数量
NUM_TEST_SAMPLES = 100                  # 合成数据测试样本数量

SEQUENCE_LENGTH = 128                   # 序列长度（时间步数 T）
FEATURE_DIM = 10                        # 特征维度（每个时间步的通道数 N）
NUM_CLASSES = 5                         # 分类类别数

SYNTHETIC_NOISE_SCALE = 0.95             # 合成数据中高斯噪声标准差（基础噪声强度）
SINE_AMPLITUDE = 0.2                    # 注入到特定通道的正弦扰动幅值（用于区分类信号）
COSINE_AMPLITUDE = 0.25                  # 注入到特定通道的余弦扰动幅值（用于区分类信号）
CLASS_OFFSET_SCALE = 0.1                # 每个样本按类别的全局偏移量比例（增加类间均值差异）

DEFAULT_WEIGHT_DECAY = 1e-4             # 优化器权重衰减（L2 正则化强度）
TRAINING_SLEEP_SECONDS = 0.08           # 每个 epoch 日志间短暂停顿（仅用于 UI 展示，不影响训练结果）

EPOCHS_RANGE = (1, 500)                 # UI: epochs 输入范围 (min, max)
BATCH_SIZE_RANGE = (8, 256)             # UI: batch size 输入范围 (min, max)
LR_RANGE = (0.0001, 0.1)                # UI: learning rate 输入范围 (min, max)


class FCNClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FCNClassifier, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=8, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv_block(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        out = self.classifier(x)
        return out


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        _, predicted = torch.max(outputs, dim=1)
        correct += (predicted == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        outputs = model(x)
        loss = criterion(outputs, y)
        total_loss += loss.item() * x.size(0)
        _, predicted = torch.max(outputs, dim=1)
        correct += (predicted == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


class ModelTrainingWorker(QThread):
    log_signal = Signal(str)
    finished_signal = Signal(float)
    error_signal = Signal(str)
    history_signal = Signal(list, list, list, list)

    def __init__(self, epochs, batch_size, lr):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

    def _generate_synthetic_data(self, num_samples, T, N, num_classes):
        X = torch.randn(num_samples, T, N) * SYNTHETIC_NOISE_SCALE
        y = torch.randint(0, num_classes, (num_samples,))

        t = torch.linspace(0, 4 * np.pi, T)
        for i in range(num_samples):
            c = y[i].item()
            X[i, :, c % N] += torch.sin(t * (c + 1)) * SINE_AMPLITUDE
            X[i, :, (c + 1) % N] += torch.cos(t * (c + 2)) * COSINE_AMPLITUDE
            X[i, :, :] += c * CLASS_OFFSET_SCALE

        return X, y

    def run(self):
        try:
            seed_torch(DEFAULT_SEED)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.log_signal.emit(f"准备开始训练，使用设备: {device}")

            num_train = NUM_TRAIN_SAMPLES
            num_test = NUM_TEST_SAMPLES
            T = SEQUENCE_LENGTH
            N = FEATURE_DIM
            num_classes = NUM_CLASSES

            X_train, y_train = self._generate_synthetic_data(num_train, T, N, num_classes)
            X_test, y_test = self._generate_synthetic_data(num_test, T, N, num_classes)

            train_dataset = TensorDataset(X_train, y_train)
            test_dataset = TensorDataset(X_test, y_test)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)

            model = FCNClassifier(input_dim=N, num_classes=num_classes).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=DEFAULT_WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

            best_acc = 0.0
            train_losses, test_losses = [], []
            train_accs, test_accs = [], []

            for epoch in range(self.epochs):
                start_time = time.time()
                train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
                test_loss, test_acc = evaluate(model, test_loader, criterion, device)
                scheduler.step()
                end_time = time.time()

                train_losses.append(train_loss)
                test_losses.append(test_loss)
                train_accs.append(train_acc)
                test_accs.append(test_acc)

                if test_acc > best_acc:
                    best_acc = test_acc

                log_str = (f"Epoch [{epoch + 1:03d}/{self.epochs}] "
                           f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                           f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | "
                           f"Time: {end_time - start_time:.2f}s")
                self.log_signal.emit(log_str)
                time.sleep(TRAINING_SLEEP_SECONDS)

            self.log_signal.emit(f"训练完成! 最佳测试准确率: {best_acc:.4f}")
            self.history_signal.emit(train_losses, test_losses, train_accs, test_accs)
            self.finished_signal.emit(best_acc)
        except Exception as e:
            self.error_signal.emit(str(e))


class ErrorClassificationWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._epochs_input = None
        self._batch_size_input = None
        self._lr_input = None
        self._log_output = None
        self._tabs = None
        self._figure = None
        self._canvas = None
        self._run_btn = None
        self._status_label = None
        self._worker = None
        self._build_ui()

    def _build_ui(self):
        self.setObjectName("errorClassRoot")
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(14, 14, 14, 14)
        main_layout.setSpacing(12)

        content_row = QHBoxLayout()
        content_row.setSpacing(12)

        left_panel = QGroupBox("网络超参数配置")
        left_layout = QFormLayout(left_panel)
        left_layout.setVerticalSpacing(12)

        self._epochs_input = QSpinBox()
        self._epochs_input.setRange(*EPOCHS_RANGE)
        self._epochs_input.setValue(DEFAULT_EPOCHS)
        left_layout.addRow("训练轮次 (Epochs):", self._epochs_input)

        self._batch_size_input = QSpinBox()
        self._batch_size_input.setRange(*BATCH_SIZE_RANGE)
        self._batch_size_input.setSingleStep(8)
        self._batch_size_input.setValue(DEFAULT_BATCH_SIZE)
        left_layout.addRow("批次大小 (Batch Size):", self._batch_size_input)

        self._lr_input = QDoubleSpinBox()
        self._lr_input.setDecimals(4)
        self._lr_input.setRange(*LR_RANGE)
        self._lr_input.setSingleStep(0.001)
        self._lr_input.setValue(DEFAULT_LR)
        left_layout.addRow("学习率 (Learning Rate):", self._lr_input)

        actions = QHBoxLayout()
        actions.addStretch()
        self._run_btn = QPushButton("开始模型训练与分类模拟")
        self._run_btn.clicked.connect(self._start_training)
        actions.addWidget(self._run_btn)
        actions.addStretch()
        left_layout.addRow("", actions)

        right_panel = QGroupBox("训练监控面板")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(8)

        self._tabs = QTabWidget()

        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        log_layout.setContentsMargins(0, 0, 0, 0)
        self._log_output = QTextEdit()
        self._log_output.setReadOnly(True)
        self._log_output.setPlaceholderText("点击左侧按钮开始模型训练，此处将输出Epoch日志。")
        log_layout.addWidget(self._log_output)
        self._tabs.addTab(log_tab, "训练日志")

        chart_tab = QWidget()
        chart_layout = QVBoxLayout(chart_tab)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        self._figure = Figure(dpi=100)
        self._figure.patch.set_facecolor('#1a2635')
        self._canvas = FigureCanvas(self._figure)
        chart_layout.addWidget(self._canvas)
        self._tabs.addTab(chart_tab, "训练曲线")

        right_layout.addWidget(self._tabs)

        content_row.addWidget(left_panel, 1)
        content_row.addWidget(right_panel, 2)
        main_layout.addLayout(content_row, 1)

        status_bar = QFrame()
        status_bar.setObjectName("errorClassStatusBar")
        status_layout = QHBoxLayout(status_bar)
        status_layout.setContentsMargins(10, 2, 10, 2)
        self._status_label = QLabel("状态：待命")
        status_layout.addWidget(self._status_label)
        status_layout.addStretch()
        main_layout.addWidget(status_bar)

        self.setStyleSheet(
            """
            QWidget#errorClassRoot {
                background: #1a2635;
            }
            QGroupBox {
                border: 1px solid rgba(123, 167, 210, 0.35);
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 14px;
                background: rgba(31, 49, 70, 0.88);
                color: #d4e8ff;
                font-size: 16px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 6px;
                color: #d9ecff;
            }
            QLabel {
                color: #d2e5fb;
                font-size: 14px;
            }
            QSpinBox, QDoubleSpinBox, QTextEdit {
                border: 1px solid rgba(143, 182, 220, 0.35);
                border-radius: 5px;
                background: rgba(21, 35, 52, 0.95);
                color: #e7f2ff;
                min-height: 28px;
                padding: 2px 8px;
            }
            QPushButton {
                border: 1px solid rgba(101, 175, 235, 0.52);
                border-radius: 8px;
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 rgba(35, 90, 132, 0.95),
                    stop: 1 rgba(31, 71, 112, 0.95)
                );
                color: #dff2ff;
                padding: 7px 14px;
                font-weight: 700;
            }
            QPushButton:hover {
                background: rgba(62, 122, 174, 0.95);
            }
            QPushButton:disabled {
                color: #87a2bd;
                background: rgba(33, 55, 77, 0.75);
            }
            QFrame#errorClassStatusBar {
                border: 1px solid rgba(126, 168, 208, 0.35);
                border-radius: 6px;
                background: rgba(25, 38, 55, 0.95);
            }
            QTabWidget::pane {
                border: 1px solid rgba(123, 167, 210, 0.35);
                border-radius: 4px;
                background: rgba(21, 35, 52, 0.95);
            }
            QTabBar::tab {
                background: rgba(31, 49, 70, 0.6);
                border: 1px solid rgba(123, 167, 210, 0.35);
                border-bottom-color: transparent;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                min-width: 100px;
                padding: 6px 12px;
                color: #87a2bd;
                font-size: 14px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: rgba(21, 35, 52, 0.95);
                color: #d4e8ff;
                border-bottom: 2px solid #63b9ff;
            }
            """
        )

    def _start_training(self):
        epochs = self._epochs_input.value()
        batch_size = self._batch_size_input.value()
        lr = self._lr_input.value()

        self._log_output.clear()
        self._figure.clear()
        self._canvas.draw()
        self._tabs.setCurrentIndex(0)
        self._run_btn.setEnabled(False)
        self._status_label.setText("状态：正在训练模型...")

        self._worker = ModelTrainingWorker(epochs, batch_size, lr)
        self._worker.log_signal.connect(self._append_log)
        self._worker.history_signal.connect(self._on_history_received)
        self._worker.finished_signal.connect(self._on_training_finished)
        self._worker.error_signal.connect(self._on_training_error)
        self._worker.start()

    def _append_log(self, text):
        self._log_output.append(text)
        scrollbar = self._log_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_history_received(self, train_losses, test_losses, train_accs, test_accs):
        self._figure.clear()

        ax1 = self._figure.add_subplot(211)
        ax1.set_facecolor('#152334')
        ax1.plot(train_losses, label='Train Loss', color='#63b9ff', linewidth=2)
        ax1.plot(test_losses, label='Test Loss', color='#ff7b72', linewidth=2)
        ax1.set_title("Loss Curve", color='#d4e8ff', fontsize=12)
        ax1.legend(facecolor='#1f3146', edgecolor='#466385', labelcolor='#d4e8ff')
        self._style_ax(ax1)

        ax2 = self._figure.add_subplot(212)
        ax2.set_facecolor('#152334')
        ax2.plot(train_accs, label='Train Acc', color='#3fb950', linewidth=2)
        ax2.plot(test_accs, label='Test Acc', color='#d2a8ff', linewidth=2)
        ax2.set_title("Accuracy Curve", color='#d4e8ff', fontsize=12)
        ax2.legend(facecolor='#1f3146', edgecolor='#466385', labelcolor='#d4e8ff')
        self._style_ax(ax2)

        self._figure.tight_layout()
        self._canvas.draw()

        self._tabs.setCurrentIndex(1)

    def _style_ax(self, ax):
        ax.tick_params(colors='#d4e8ff')
        for spine in ax.spines.values():
            spine.set_color('#466385')

    def _on_training_finished(self, best_acc):
        self._run_btn.setEnabled(True)
        self._status_label.setText(f"状态：训练完成 (最高精度: {best_acc:.2%})")

    def _on_training_error(self, err_msg):
        self._run_btn.setEnabled(True)
        self._log_output.append(f"发生错误: {err_msg}")
        self._status_label.setText("状态：训练异常中断")