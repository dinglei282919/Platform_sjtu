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
from matplotlib import font_manager

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
    QVBoxLayout,
    QWidget,
)


def seed_torch(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


DEFAULT_SEED = 42
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 0.001

NUM_TRAIN_SAMPLES = 500
NUM_TEST_SAMPLES = 100

SEQUENCE_LENGTH = 128
FEATURE_DIM = 10
NUM_CLASSES = 5

SECURITY_CLASS_NAMES = [
    "正常运行",
    "数据注入威胁",
    "拒绝服务威胁",
    "重放攻击威胁",
    "拓扑篡改威胁",
]

SYNTHETIC_NOISE_SCALE = 0.95
SINE_AMPLITUDE = 0.2
COSINE_AMPLITUDE = 0.25
CLASS_OFFSET_SCALE = 0.1

DEFAULT_WEIGHT_DECAY = 1e-4
TRAINING_SLEEP_SECONDS = 0.08

EPOCHS_RANGE = (1, 500)
BATCH_SIZE_RANGE = (8, 256)
LR_RANGE = (0.0001, 0.1)


def configure_matplotlib_chinese_font():
    # Windows/跨平台常见中文字体回退，避免中文渲染成方块
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for font_name in candidates:
        if font_name in available:
            matplotlib.rcParams["font.sans-serif"] = [font_name] + matplotlib.rcParams.get("font.sans-serif", [])
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


configure_matplotlib_chinese_font()


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
    history_signal = Signal(object, object)

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

            model.eval()
            all_true = []
            all_pred = []
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                outputs = model(x)
                _, predicted = torch.max(outputs, dim=1)
                all_true.extend(y.cpu().tolist())
                all_pred.extend(predicted.cpu().tolist())

            self.log_signal.emit(f"训练完成! 最佳测试准确率: {best_acc:.4f}")
            self.history_signal.emit(all_true, all_pred)
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

        left_panel = QGroupBox("输入与训练日志")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(12)
        left_layout.setContentsMargins(12, 16, 12, 12)

        inputs_row = QHBoxLayout()
        inputs_row.setSpacing(16)

        epochs_box = QVBoxLayout()
        epochs_box.setSpacing(6)
        epochs_box.addWidget(QLabel("Epochs"))
        self._epochs_input = QSpinBox()
        self._epochs_input.setRange(*EPOCHS_RANGE)
        self._epochs_input.setValue(DEFAULT_EPOCHS)
        self._epochs_input.setFixedWidth(90)
        epochs_box.addWidget(self._epochs_input)
        epochs_box.addStretch()
        inputs_row.addLayout(epochs_box)

        batch_box = QVBoxLayout()
        batch_box.setSpacing(6)
        batch_box.addWidget(QLabel("Batch Size"))
        self._batch_size_input = QSpinBox()
        self._batch_size_input.setRange(*BATCH_SIZE_RANGE)
        self._batch_size_input.setSingleStep(8)
        self._batch_size_input.setValue(DEFAULT_BATCH_SIZE)
        self._batch_size_input.setFixedWidth(90)
        batch_box.addWidget(self._batch_size_input)
        batch_box.addStretch()
        inputs_row.addLayout(batch_box)

        lr_box = QVBoxLayout()
        lr_box.setSpacing(6)
        lr_box.addWidget(QLabel("Learning Rate"))
        self._lr_input = QDoubleSpinBox()
        self._lr_input.setDecimals(4)
        self._lr_input.setRange(*LR_RANGE)
        self._lr_input.setSingleStep(0.001)
        self._lr_input.setValue(DEFAULT_LR)
        self._lr_input.setFixedWidth(100)
        lr_box.addWidget(self._lr_input)
        lr_box.addStretch()
        inputs_row.addLayout(lr_box)

        inputs_row.addStretch()

        btn_box = QVBoxLayout()
        btn_box.addStretch()
        self._run_btn = QPushButton("开始潜在安全威胁识别与自动分类训练")
        self._run_btn.clicked.connect(self._start_training)
        self._run_btn.setFixedHeight(36)
        self._run_btn.setMinimumWidth(180)
        btn_box.addWidget(self._run_btn)
        inputs_row.addLayout(btn_box)

        left_layout.addLayout(inputs_row)

        self._log_output = QTextEdit()
        self._log_output.setReadOnly(True)
        self._log_output.setPlaceholderText("训练日志将在这里显示。")
        self._log_output.setStyleSheet(
            "QTextEdit{background:rgba(21,35,52,0.95);color:#e7f2ff;padding:8px;border-radius:6px}")
        left_layout.addWidget(self._log_output, 1)

        content_row.addWidget(left_panel, 1)

        right_panel = QGroupBox("潜在安全威胁识别与自动分类结果")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(12)
        right_layout.setContentsMargins(12, 16, 12, 12)

        self._result_table = QTableWidget()
        self._result_table.setStyleSheet(
            "QTableWidget{background:#152334;color:#d4e8ff;gridline-color:#203445;border-radius:6px;}"
            "QTableWidget::item{padding:6px}",
        )
        self._result_table.setAlternatingRowColors(True)
        self._result_table.setShowGrid(True)
        self._result_table.setSelectionMode(QAbstractItemView.NoSelection)
        self._result_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._result_table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._result_table.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)
        self._result_table.verticalHeader().setDefaultAlignment(Qt.AlignCenter)
        right_layout.addWidget(self._result_table, 1)

        self._figure = Figure(dpi=100)
        self._figure.patch.set_facecolor('#1a2635')
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setMinimumHeight(280)
        right_layout.addWidget(self._canvas, 1)

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
            QTableWidget {
                background: #152334;
                color: #d4e8ff;
                gridline-color: #2e4a63;
            }
            QHeaderView::section {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 rgba(31,49,70,0.95), stop:1 rgba(25,38,55,0.95));
                color: #d9ecff;
                padding: 6px;
                border: none;
                font-weight: 700;
            }
            QTableWidget::item {
                padding: 6px;
            }
            """
        )

    def _start_training(self):
        epochs = self._epochs_input.value()
        batch_size = self._batch_size_input.value()
        lr = self._lr_input.value()

        self._log_output.clear()

        try:
            self._result_table.clear()
            self._result_table.setRowCount(0)
            self._result_table.setColumnCount(0)
        except Exception:
            pass

        self._figure.clear()
        self._canvas.draw()
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

    def _on_history_received(self, y_true, y_pred):

        import numpy as _np

        y_true = _np.array(y_true)
        y_pred = _np.array(y_pred)
        num_classes = NUM_CLASSES

        cm = _np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(y_true.tolist(), y_pred.tolist()):
            if 0 <= int(t) < num_classes and 0 <= int(p) < num_classes:
                cm[int(t), int(p)] += 1

        self._result_table.clear()
        self._result_table.setRowCount(num_classes + 1)
        self._result_table.setColumnCount(num_classes + 1)

        h_labels = [f"预测-{SECURITY_CLASS_NAMES[i]}" for i in range(num_classes)] + ["合计"]
        v_labels = [f"真实-{SECURITY_CLASS_NAMES[i]}" for i in range(num_classes)] + ["合计"]
        self._result_table.setHorizontalHeaderLabels(h_labels)
        self._result_table.setVerticalHeaderLabels(v_labels)

        row_sums = cm.sum(axis=1)
        col_sums = cm.sum(axis=0)
        total = cm.sum()
        for i in range(num_classes):
            for j in range(num_classes):
                item = QTableWidgetItem(str(int(cm[i, j])))
                item.setTextAlignment(Qt.AlignCenter)
                self._result_table.setItem(i, j, item)

            row_item = QTableWidgetItem(str(int(row_sums[i])))
            row_item.setTextAlignment(Qt.AlignCenter)
            self._result_table.setItem(i, num_classes, row_item)

        for j in range(num_classes):
            col_item = QTableWidgetItem(str(int(col_sums[j])))
            col_item.setTextAlignment(Qt.AlignCenter)
            self._result_table.setItem(num_classes, j, col_item)

        total_item = QTableWidgetItem(str(int(total)))
        total_item.setTextAlignment(Qt.AlignCenter)
        self._result_table.setItem(num_classes, num_classes, total_item)

        self._figure.clear()

        ax1 = self._figure.add_subplot(121)
        ax1.set_facecolor('#152334')
        im = ax1.imshow(cm, interpolation='nearest', cmap='Blues')
        ax1.set_title('潜在安全威胁识别混淆矩阵', color='#d4e8ff', fontsize=12)
        ax1.set_xlabel('预测类别', color='#d4e8ff')
        ax1.set_ylabel('真实类别', color='#d4e8ff')
        ticks = list(range(num_classes))
        ax1.set_xticks(ticks)
        ax1.set_yticks(ticks)
        ax1.set_xticklabels([SECURITY_CLASS_NAMES[i] for i in ticks], color='#d4e8ff', rotation=20, ha='right')
        ax1.set_yticklabels([SECURITY_CLASS_NAMES[i] for i in ticks], color='#d4e8ff')

        cm_max = cm.max() if cm.size and cm.max() > 0 else 1
        for i in range(num_classes):
            for j in range(num_classes):
                val = int(cm[i, j])
                txt_color = 'white' if val > cm_max / 2 else 'black'
                ax1.text(j, i, val, ha='center', va='center', color=txt_color, fontsize=10, fontweight='600')

        try:
            self._figure.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        except Exception:
            pass
        self._style_ax(ax1)

        ax2 = self._figure.add_subplot(122)
        ax2.set_facecolor('#152334')
        with _np.errstate(divide='ignore', invalid='ignore'):
            per_class_acc = _np.divide(_np.diag(cm), row_sums, out=_np.zeros_like(row_sums, dtype=float),
                                       where=row_sums != 0)
        ax2.bar(ticks, per_class_acc, color='#63b9ff')
        ax2.set_ylim(0, 1.0)
        ax2.set_xticks(ticks)
        ax2.set_xticklabels([SECURITY_CLASS_NAMES[i] for i in ticks], color='#d4e8ff', rotation=20, ha='right')
        ax2.set_ylabel('识别召回率', color='#d4e8ff')
        ax2.set_title('各类潜在安全威胁识别召回率', color='#d4e8ff', fontsize=12)

        for bar in ax2.patches:
            bar.set_edgecolor('#2e4a63')
            bar.set_linewidth(0.8)
            bar.set_alpha(0.95)
        ax2.grid(axis='y', color='#203445', linestyle='--', linewidth=0.6, alpha=0.6)
        self._style_ax(ax2)

        self._figure.tight_layout(pad=2.0)
        self._canvas.draw()

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