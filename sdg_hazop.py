# -*- coding: utf-8 -*-
"""
SDG-HAZOP 定量风险分析 (PySide6 版本)
左侧面板深色背景，所有控件带灰白色边框，统一风格
"""
import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QComboBox, QLineEdit, QTextEdit, QFrame,
    QGroupBox, QFormLayout, QMessageBox
)
from PySide6.QtCore import Qt, Signal
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Optional

# ================================================================
# 算法核心 (保持不变)
# ================================================================
class NodeType(Enum):
    CAUSE = "R"
    PARAMETER = "P"
    CONSEQUENCE = "C"

class EdgeType(Enum):
    INCREMENT = "+"
    DECREMENT = "-"

@dataclass
class SDGNode:
    id: str
    name: str
    node_type: NodeType
    base_probability: Optional[float] = None

@dataclass
class SDGEdge:
    source: str
    target: str
    edge_type: EdgeType
    conditional_prob: float = 0.5

class FuzzyExpertEvaluator:
    FUZZY_TERMS = {
        "很小": (0.0, 0.0, 0.1, 0.2), "小": (0.1, 0.2, 0.3, 0.3),
        "较小": (0.2, 0.3, 0.4, 0.5), "中等": (0.4, 0.5, 0.6, 0.6),
        "较大": (0.5, 0.6, 0.7, 0.8), "大": (0.7, 0.8, 0.9, 0.9),
        "很大": (0.8, 0.9, 1.0, 1.0),
    }
    def __init__(self, expert_weights: Dict[str, float] = None):
        if expert_weights is None:
            expert_weights = {"专家A": 0.85, "专家B": 0.90, "专家C": 0.75}
        self.expert_weights = expert_weights

    def _fps_left_right(self, fuzzy_num, lam=0.5):
        a,b,c,d = fuzzy_num
        L_lam = a + lam*(b-a)
        R_lam = d - lam*(d-c)
        return (1.0 - L_lam), R_lam

    @staticmethod
    def _fps_to_ffr(fps):
        if fps <= 0.0: return 1.0
        if fps >= 1.0: return 0.0
        k = ((1.0 - fps) / fps) ** (1.0/3.0) * 2.301
        return 1.0 / (10.0 ** k)

    def evaluate_term(self, term: str, is_frequency: bool = True) -> float:
        if term not in self.FUZZY_TERMS:
            raise ValueError(f"未知术语: {term}")
        fuzzy_num = self.FUZZY_TERMS[term]
        fps_l, fps_r = self._fps_left_right(fuzzy_num, lam=0.5)
        fps = (fps_r + 1.0 - fps_l) / 2.0
        return self._fps_to_ffr(fps) if is_frequency else fps

    def evaluate_group(self, opinions: Dict[str, str], is_frequency: bool = True) -> float:
        total_weight = 0.0
        weighted_fps = 0.0
        for expert, term in opinions.items():
            weight = self.expert_weights.get(expert, 0.5)
            if term not in self.FUZZY_TERMS: continue
            fuzzy_num = self.FUZZY_TERMS[term]
            fps_l, fps_r = self._fps_left_right(fuzzy_num, lam=0.5)
            fps = (fps_r + 1.0 - fps_l) / 2.0
            weighted_fps += weight * fps
            total_weight += weight
        if total_weight == 0: return 0.0
        avg_fps = weighted_fps / total_weight
        return self._fps_to_ffr(avg_fps) if is_frequency else avg_fps

class ProbabilisticSDG:
    def __init__(self):
        self.nodes: Dict[str, SDGNode] = {}
        self.edges: Dict[str, SDGEdge] = {}
        self.adj: Dict[str, List[str]] = {}
        self.rev_adj: Dict[str, List[str]] = {}

    def add_node(self, node: SDGNode):
        self.nodes[node.id] = node
        self.adj.setdefault(node.id, [])
        self.rev_adj.setdefault(node.id, [])

    def add_edge(self, edge: SDGEdge):
        key = f"{edge.source}->{edge.target}"
        self.edges[key] = edge
        self.adj.setdefault(edge.source, []).append(edge.target)
        self.rev_adj.setdefault(edge.target, []).append(edge.source)

    def get_cond_prob(self, src: str, tgt: str) -> Optional[float]:
        key = f"{src}->{tgt}"
        return self.edges[key].conditional_prob if key in self.edges else None

    def calculate_path_probability(self, path: List[str]) -> Tuple[float, List[str]]:
        if len(path) < 2:
            return 0.0, ["路径过短"]
        start_node = self.nodes.get(path[0])
        if start_node is None or start_node.base_probability is None:
            raise ValueError(f"起始节点 {path[0]} 缺少基础概率")
        prob = start_node.base_probability
        steps = [f"P({path[0]}) = {prob:.6f} (基础频率)"]
        for i in range(len(path)-1):
            src, tgt = path[i], path[i+1]
            cp = self.get_cond_prob(src, tgt)
            if cp is None:
                raise ValueError(f"缺少条件概率: {src}->{tgt}")
            old_prob = prob
            prob *= cp
            steps.append(f"  × P({tgt}|{src}) = {cp:.4f}  →  {old_prob:.6f} × {cp:.4f} = {prob:.8f}")
        return prob, steps

    def calculate_or_probability(self, probs: List[float]) -> Tuple[float, List[str]]:
        if not probs: return 0.0, []
        result = 0.0
        steps = []
        for i, p in enumerate(probs):
            old = result
            result = result + p - result * p
            steps.append(f"步骤{i+1}: P_total = {old:.6f} + {p:.6f} - {old:.6f}×{p:.6f} = {result:.8f}")
        return result, steps

    def forward_reasoning(self, start_id: str, max_depth: int = 10) -> List[List[str]]:
        all_paths = []
        def dfs(current, path, depth):
            if depth > max_depth: return
            neighbors = self.adj.get(current, [])
            unvisited = [n for n in neighbors if n not in path]
            if not unvisited:
                all_paths.append(path.copy())
                return
            for nxt in unvisited:
                path.append(nxt)
                dfs(nxt, path, depth+1)
                path.pop()
        dfs(start_id, [start_id], 0)
        return all_paths

    def backward_reasoning(self, target_id: str) -> List[List[str]]:
        all_paths = []
        def dfs(current, path):
            preds = self.rev_adj.get(current, [])
            unvisited = [p for p in preds if p not in path]
            if not unvisited:
                all_paths.append(path.copy())
                return
            for p in unvisited:
                path.append(p)
                dfs(p, path)
                path.pop()
        dfs(target_id, [target_id])
        return all_paths

class RiskMatrix:
    PROB_LEVELS = [
        (1, "极低", 1e-5, 1e-3), (2, "低", 1e-3, 1e-2),
        (3, "中等", 1e-2, 1e-1), (4, "高", 1e-1, 1.0),
        (5, "极高", 1.0, float('inf'))
    ]
    SEV_MAP = {
        "爆炸": (4, "灾难性"), "火灾": (3, "严重"),
        "泄漏": (3, "严重"), "停车": (2, "一般"),
        "起跳": (2, "一般"), "报警": (1, "轻微")
    }

    @classmethod
    def get_prob_level(cls, prob):
        for level, desc, low, high in cls.PROB_LEVELS:
            if low <= prob < high:
                return level, desc
        return 5, "极高"

    @classmethod
    def get_sev_level(cls, node_name):
        for key, (level, desc) in cls.SEV_MAP.items():
            if key in node_name:
                return level, desc
        return 1, "轻微"

    @classmethod
    def get_risk(cls, prob_level, sev_level):
        matrix = [
            ["低", "低", "中", "中"],
            ["低", "中", "中", "高"],
            ["中", "中", "高", "极高"],
            ["中", "高", "极高", "极高"],
            ["高", "极高", "极高", "极高"]
        ]
        row = min(prob_level-1, 4)
        col = min(sev_level-1, 3)
        level = matrix[row][col]
        actions = {
            "低": "可接受风险，常规管理",
            "中": "可容忍风险，建议监控",
            "高": "不可接受风险，需增加保护层",
            "极高": "重大风险，立即整改"
        }
        return level, actions.get(level, "待评估")

# ================================================================
# PySide6 Widget (深色背景 + 灰白边框控件)
# ================================================================
class SDG_HazopWidget(QWidget):
    request_sil_validation = Signal(str, float, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.nodes_data = []
        self.edges_data = []
        self.G = nx.DiGraph()
        self.pos = None
        self.sis_required_nodes = []
        self.consequence_probs = {}
        self._init_ui()
        self._load_example_te()

    def _init_ui(self):
        # ========== 定义统一样式 ==========
        # 按钮样式（灰白边框）
        btn_style = """
            QPushButton {
                background: rgba(62, 109, 165, 0.3);
                border: 1px solid rgba(200, 200, 200, 0.3);
                border-radius: 6px;
                padding: 6px 12px;
                color: #d8e7ff;
            }
            QPushButton:hover {
                background: rgba(99, 157, 230, 0.4);
            }
        """
        # 输入框样式（灰白边框）
        input_style = """
            QLineEdit, QComboBox {
                background: rgba(13, 31, 59, 0.8);
                border: 1px solid rgba(200, 200, 200, 0.3);
                border-radius: 4px;
                padding: 4px;
                color: #d8e7ff;
            }
            QLineEdit:focus, QComboBox:focus {
                border-color: rgba(123, 176, 247, 0.8);
            }
        """
        # GroupBox 样式
        groupbox_style = """
            QGroupBox {
                border: 1px solid rgba(123, 176, 247, 0.3);
                border-radius: 8px;
                margin-top: 1ex;
                color: #d8e7ff;
                background: rgba(13, 31, 59, 0.8);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #c3d8f6;
            }
        """
        # ==================================

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # 左侧控制面板 - 深色背景
        left_panel = QFrame()
        left_panel.setFrameStyle(QFrame.StyledPanel)
        left_panel.setMinimumWidth(380)
        left_panel.setStyleSheet("background-color: rgba(13, 31, 59, 0.95); border: none;")
        left_layout = QVBoxLayout(left_panel)

        # 添加节点组
        node_group = QGroupBox("添加节点")
        node_group.setStyleSheet(groupbox_style)
        node_form = QFormLayout()
        self.id_edit = QLineEdit("R1")
        self.id_edit.setStyleSheet(input_style)
        self.name_edit = QLineEdit("冷却水故障")
        self.name_edit.setStyleSheet(input_style)
        self.type_combo = QComboBox()
        self.type_combo.addItems(["原因 (R)", "参数 (P)", "后果 (C)"])
        self.type_combo.setStyleSheet(input_style)
        self.prob_edit = QLineEdit("0.018")
        self.prob_edit.setStyleSheet(input_style)
        self.fuzzy_combo = QComboBox()
        self.fuzzy_combo.addItems(["很小","小","较小","中等","较大","大","很大"])
        self.fuzzy_combo.setCurrentText("中等")
        self.fuzzy_combo.setStyleSheet(input_style)

        node_form.addRow("ID:", self.id_edit)
        node_form.addRow("名称:", self.name_edit)
        node_form.addRow("类型:", self.type_combo)
        node_form.addRow("概率/频率:", self.prob_edit)
        node_form.addRow("模糊术语:", self.fuzzy_combo)
        btn_apply_fuzzy = QPushButton("应用模糊→概率")
        btn_apply_fuzzy.setStyleSheet(btn_style)
        btn_apply_fuzzy.clicked.connect(self._apply_fuzzy)
        node_form.addRow(btn_apply_fuzzy)
        btn_add_node = QPushButton("➕ 添加节点")
        btn_add_node.setStyleSheet(btn_style)
        btn_add_node.clicked.connect(self._add_node)
        node_form.addRow(btn_add_node)
        node_group.setLayout(node_form)
        left_layout.addWidget(node_group)

        # 添加边组
        edge_group = QGroupBox("添加因果关系边")
        edge_group.setStyleSheet(groupbox_style)
        edge_form = QFormLayout()
        self.src_combo = QComboBox()
        self.src_combo.setStyleSheet(input_style)
        self.tgt_combo = QComboBox()
        self.tgt_combo.setStyleSheet(input_style)
        self.edge_type_combo = QComboBox()
        self.edge_type_combo.addItems(["增量 (+)", "减量 (-)"])
        self.edge_type_combo.setStyleSheet(input_style)
        self.edge_prob_edit = QLineEdit("0.85")
        self.edge_prob_edit.setStyleSheet(input_style)
        edge_form.addRow("源节点:", self.src_combo)
        edge_form.addRow("目标节点:", self.tgt_combo)
        edge_form.addRow("影响类型:", self.edge_type_combo)
        edge_form.addRow("条件概率:", self.edge_prob_edit)
        btn_add_edge = QPushButton("🔗 添加边")
        btn_add_edge.setStyleSheet(btn_style)
        btn_add_edge.clicked.connect(self._add_edge)
        edge_form.addRow(btn_add_edge)
        edge_group.setLayout(edge_form)
        left_layout.addWidget(edge_group)

        # 操作按钮
        btn_load_te = QPushButton("📋 加载TE示例")
        btn_load_te.setStyleSheet(btn_style)
        btn_load_te.clicked.connect(self._load_example_te)
        btn_clear = QPushButton("🗑️ 清空所有")
        btn_clear.setStyleSheet(btn_style)
        btn_clear.clicked.connect(self._clear_all)
        btn_refresh = QPushButton("🔄 刷新视图")
        btn_refresh.setStyleSheet(btn_style)
        btn_refresh.clicked.connect(self._update_drawing)
        btn_run = QPushButton("🚀 运行完整定量分析")
        btn_run.setStyleSheet(btn_style + "font-weight:bold;")
        btn_run.clicked.connect(self._run_analysis)

        left_layout.addWidget(btn_load_te)
        left_layout.addWidget(btn_clear)
        left_layout.addWidget(btn_refresh)
        left_layout.addWidget(btn_run)

        self.status_label = QLabel("节点: 0 | 边: 0 | SIS需求: 0")
        left_layout.addWidget(self.status_label)
        left_layout.addStretch()

        # 右侧绘图+日志
        right_panel = QFrame()
        right_panel.setStyleSheet("background-color: rgba(13, 31, 59, 0.7); border: none;")
        right_layout = QVBoxLayout(right_panel)

        self.fig = Figure(figsize=(7,5))
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.cid = self.canvas.mpl_connect('button_press_event', self._on_click)
        right_layout.addWidget(self.canvas, 1)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(150)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background: rgba(13, 31, 59, 0.8);
                border: 1px solid rgba(200,200,200,0.2);
                border-radius: 4px;
                color: #d8e7ff;
            }
        """)
        right_layout.addWidget(self.log_text)

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, 1)

        self._update_combos()
        self._update_drawing()

    # ---------- 以下所有方法与之前完全相同（只移除了字体/背景设置） ----------
    def _apply_fuzzy(self):
        term = self.fuzzy_combo.currentText()
        if term:
            try:
                evaluator = FuzzyExpertEvaluator()
                prob = evaluator.evaluate_term(term, is_frequency=True)
                self.prob_edit.setText(f"{prob:.6f}")
                self._log(f"模糊术语 '{term}' → 概率 {prob:.6f} 次/年")
            except Exception as e:
                QMessageBox.warning(self, "错误", str(e))

    def _add_node(self):
        nid = self.id_edit.text().strip()
        name = self.name_edit.text().strip()
        type_str = self.type_combo.currentText()
        prob_str = self.prob_edit.text().strip()
        if not nid or not name:
            QMessageBox.warning(self, "错误", "ID和名称不能为空")
            return
        if any(n[0] == nid for n in self.nodes_data):
            QMessageBox.warning(self, "错误", f"ID '{nid}' 已存在")
            return
        ntype = "R" if "原因" in type_str else "C" if "后果" in type_str else "P"
        prob = 0.0
        if prob_str and ntype == "R":
            try:
                prob = float(prob_str)
            except:
                QMessageBox.warning(self, "错误", "概率必须是数字")
                return
        self.nodes_data.append((nid, name, ntype, prob))
        self.G.add_node(nid, name=name, type=ntype, prob=prob)
        self.id_edit.setText("")
        self.name_edit.setText("")
        self.prob_edit.setText("0.01")
        self._update_combos()
        self._update_drawing()
        self._update_status()
        self._log(f"节点添加: {nid} ({name}) [类型:{ntype}, 概率:{prob}]")

    def _add_edge(self):
        src = self.src_combo.currentText()
        tgt = self.tgt_combo.currentText()
        edge_type_str = self.edge_type_combo.currentText()
        prob_str = self.edge_prob_edit.text().strip()
        if not src or not tgt:
            QMessageBox.warning(self, "错误", "请选择源和目标节点")
            return
        if src == tgt:
            QMessageBox.warning(self, "错误", "不能连接自己")
            return
        if any(e[0]==src and e[1]==tgt for e in self.edges_data):
            QMessageBox.warning(self, "错误", "该边已存在")
            return
        etype = "+" if "增量" in edge_type_str else "-"
        cp = 0.5
        if prob_str:
            try:
                cp = float(prob_str)
                if not (0 <= cp <= 1):
                    raise ValueError
            except:
                QMessageBox.warning(self, "错误", "条件概率必须是 0~1 的数字")
                return
        self.edges_data.append((src, tgt, etype, cp))
        self.G.add_edge(src, tgt, type=etype, prob=cp)
        self._update_combos()
        self._update_drawing()
        self._update_status()
        self._log(f"边添加: {src} -> {tgt} [{etype}, P={cp}]")

    def _clear_all(self):
        self.nodes_data.clear()
        self.edges_data.clear()
        self.G.clear()
        self.sis_required_nodes.clear()
        self.consequence_probs.clear()
        self.log_text.clear()
        self._update_combos()
        self._update_drawing()
        self._update_status()
        self._log("已清空所有数据")

    def _update_combos(self):
        ids = [n[0] for n in self.nodes_data]
        self.src_combo.clear()
        self.tgt_combo.clear()
        self.src_combo.addItems(ids)
        self.tgt_combo.addItems(ids)
        if ids:
            self.src_combo.setCurrentIndex(0)
            self.tgt_combo.setCurrentIndex(len(ids)-1 if len(ids)>1 else 0)

    def _update_status(self):
        self.status_label.setText(f"节点: {len(self.nodes_data)} | 边: {len(self.edges_data)} | SIS需求: {len(self.sis_required_nodes)}")

    def _update_drawing(self):
        self.ax.clear()
        if len(self.G.nodes) == 0:
            self.ax.text(0.5, 0.5, "请添加节点和边", ha='center', va='center', fontsize=14, color='gray')
            self.canvas.draw()
            return

        self.pos = nx.spring_layout(self.G, seed=42, k=1.8)
        normal_nodes = [n for n in self.G.nodes if n not in self.sis_required_nodes]
        sis_nodes = [n for n in self.G.nodes if n in self.sis_required_nodes]

        if normal_nodes:
            colors = ['#ff7f7f' if self.G.nodes[n].get('type','P')=='R'
                      else '#7fbfff' if self.G.nodes[n].get('type','P')=='C'
                      else '#7fff7f' for n in normal_nodes]
            nx.draw_networkx_nodes(self.G, self.pos, nodelist=normal_nodes, node_color=colors,
                                   node_size=1800, ax=self.ax, edgecolors='black', linewidths=1.5)
        if sis_nodes:
            colors = ['#ff7f7f' if self.G.nodes[n].get('type','P')=='R'
                      else '#7fbfff' if self.G.nodes[n].get('type','P')=='C'
                      else '#7fff7f' for n in sis_nodes]
            nx.draw_networkx_nodes(self.G, self.pos, nodelist=sis_nodes, node_color=colors,
                                   node_size=2000, ax=self.ax, edgecolors='red', linewidths=3.0)

        labels = {n: f"{n}\n{self.G.nodes[n].get('name','')}" for n in self.G.nodes}
        nx.draw_networkx_labels(self.G, self.pos, labels=labels, font_size=8, ax=self.ax)

        e_colors = ['red' if self.G.edges[u,v].get('type','+')=='-' else 'black' for u,v in self.G.edges]
        nx.draw_networkx_edges(self.G, self.pos, edge_color=e_colors, arrowstyle='-|>', arrowsize=20, width=2.0, ax=self.ax)
        e_labels = {(u,v): f"P={d.get('prob',0.5):.2f}" for u,v,d in self.G.edges(data=True)}
        nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=e_labels, font_size=7, ax=self.ax, label_pos=0.3)

        if sis_nodes:
            self.ax.text(0.02, 0.98, "● 红色边框 = 需要SIS验证的节点", transform=self.ax.transAxes,
                         fontsize=9, color='red', verticalalignment='top',
                         bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
        self.ax.set_title("SDG 因果模型 (红R:原因, 绿P:参数, 蓝C:后果)")
        self.ax.axis('off')
        self.canvas.draw()

    def _on_click(self, event):
        if event.inaxes != self.ax or self.pos is None:
            return
        click_pos = np.array([event.xdata, event.ydata])
        min_dist = float('inf')
        closest_node = None
        for node, pos in self.pos.items():
            dist = np.linalg.norm(click_pos - np.array(pos))
            if dist < min_dist:
                min_dist = dist
                closest_node = node
        if closest_node is not None and min_dist < 0.15:
            if closest_node in self.sis_required_nodes:
                prob = self.consequence_probs.get(closest_node, 0.001)
                sev = self.consequence_probs.get(closest_node+'_sev', 3)
                self._log(f"双击节点 {closest_node} 触发 SIL 验证")
                self.request_sil_validation.emit(closest_node, prob, sev)

    def _log(self, msg):
        self.log_text.append(msg)

    def _load_example_te(self):
        self._clear_all()
        example_nodes = [("R1","冷却水故障","R",0.018), ("R2","搅拌器故障","R",0.008),
                         ("T1","反应器温度","P",0.0), ("P1","反应器压力","P",0.0),
                         ("C1","反应器爆炸","C",0.0), ("C2","安全阀起跳","C",0.0)]
        for nid,name,typ,prob in example_nodes:
            self.nodes_data.append((nid,name,typ,prob))
            self.G.add_node(nid, name=name, type=typ, prob=prob)
        example_edges = [("R1","T1","+",0.85), ("R2","T1","+",0.70),
                         ("T1","P1","+",0.90), ("P1","C1","+",0.65), ("P1","C2","+",0.40)]
        for src,tgt,etyp,cp in example_edges:
            self.edges_data.append((src,tgt,etyp,cp))
            self.G.add_edge(src,tgt, type=etyp, prob=cp)
        self._update_combos()
        self._update_drawing()
        self._update_status()
        self._log("✅ 已加载 TE 过程反应器超压示例模型")

    def _run_analysis(self):
        self.log_text.clear()
        self._log("="*70)
        self._log("  SDG-HAZOP 完整定量风险分析报告")
        self._log("="*70)

        if len(self.nodes_data) < 2:
            self._log("错误: 节点数不足")
            return

        sdg = ProbabilisticSDG()
        for nid,name,typ,prob in self.nodes_data:
            nt = NodeType.CAUSE if typ=='R' else NodeType.CONSEQUENCE if typ=='C' else NodeType.PARAMETER
            sdg.add_node(SDGNode(nid,name,nt,prob if typ=='R' else None))
        for src,tgt,etyp,cp in self.edges_data:
            et = EdgeType.INCREMENT if etyp=='+' else EdgeType.DECREMENT
            sdg.add_edge(SDGEdge(src,tgt,et,cp))

        cause_nodes = [nid for nid,_,typ,_ in self.nodes_data if typ=='R']
        conseq_nodes = [nid for nid,_,typ,_ in self.nodes_data if typ=='C']
        if not cause_nodes:
            self._log("错误: 没有原因节点")
            return

        self.consequence_probs = {}
        self.sis_required_nodes.clear()

        # 正向推理
        self._log("\n--- 正向推理: 风险路径与概率计算 ---")
        for r in cause_nodes:
            paths = sdg.forward_reasoning(r)
            if not paths:
                self._log(f"节点 {r} 无任何传播路径")
                continue
            for p in paths:
                try:
                    prob, steps = sdg.calculate_path_probability(p)
                    self._log(f"\n路径: {' → '.join(p)}")
                    for step in steps:
                        self._log(f"  {step}")
                    self._log(f"  >> 最终概率 = {prob:.8f} 次/年")
                except Exception as e:
                    self._log(f"路径 {' → '.join(p)} 计算失败: {e}")

        # 后果节点总概率
        self._log("\n--- 后果节点总概率计算 (并联/或门聚合) ---")
        for c in conseq_nodes:
            paths = sdg.backward_reasoning(c)
            if not paths:
                continue
            path_probs = []
            for p in paths:
                rev = list(reversed(p))
                try:
                    prob, _ = sdg.calculate_path_probability(rev)
                    path_probs.append(prob)
                except:
                    pass

            if len(path_probs) == 1:
                total = path_probs[0]
                self._log(f"\n后果 {c}: 仅有1条路径 → 总概率 = {total:.8f}")
            elif len(path_probs) > 1:
                total, agg_steps = sdg.calculate_or_probability(path_probs)
                self._log(f"\n后果 {c}: 汇聚 {len(path_probs)} 条路径 (或门)")
                for i, p in enumerate(path_probs):
                    self._log(f"  路径{i+1}概率: {p:.8f}")
                self._log(f"  聚合计算:")
                for st in agg_steps:
                    self._log(f"    {st}")
                self._log(f"  >> 总概率 = {total:.8f} 次/年")
            else:
                total = 0.0

            self.consequence_probs[c] = total
            p_level, p_desc = RiskMatrix.get_prob_level(total)
            s_level, s_desc = RiskMatrix.get_sev_level(c)
            risk_lvl, action = RiskMatrix.get_risk(p_level, s_level)
            self.consequence_probs[c+'_sev'] = s_level

            self._log(f"\n  【风险矩阵定级】")
            self._log(f"    可能性等级: P{p_level} ({p_desc})")
            self._log(f"    严重性等级: S{s_level} ({s_desc})")
            self._log(f"    ★ 综合风险: {risk_lvl}")
            self._log(f"    建议措施: {action}")

            # LOPA
            if total > 0:
                pfd_dcs = 0.1
                pfd_sv = 0.01
                residual = total * pfd_dcs * pfd_sv
                tolerance = 1e-6
                self._log(f"\n  【LOPA 保护层分析】")
                self._log(f"    假设保护层: DCS(PFD={pfd_dcs}) + 安全阀(PFD={pfd_sv})")
                self._log(f"    原始风险: {total:.8f} 次/年")
                self._log(f"    残余风险: {residual:.8f} 次/年")
                self._log(f"    可容忍标准: {tolerance} 次/年")
                if residual <= tolerance:
                    self._log("    ✅ 残余风险可接受")
                else:
                    rrf = residual / tolerance
                    self._log(f"    ❌ 需要新增SIF，RRF={rrf:.2f}")
                    if rrf >= 10:
                        if rrf < 100: sil = 1
                        elif rrf < 1000: sil = 2
                        elif rrf < 10000: sil = 3
                        else: sil = 4
                        self._log(f"    → 目标 SIL 等级: {sil}")
                        if c not in self.sis_required_nodes:
                            self.sis_required_nodes.append(c)

        # 反向推理
        if conseq_nodes:
            self._log("\n--- 反向推理: 后果原因追溯 ---")
            for c in conseq_nodes:
                paths = sdg.backward_reasoning(c)
                self._log(f"\n后果 {c} 的可能原因链:")
                for p in paths:
                    rev = list(reversed(p))
                    self._log(f"  {' → '.join(rev)}")

        self._log("\n"+"="*70)
        self._log("  分析完成")
        self._update_drawing()
        self._update_status()
        if self.sis_required_nodes:
            self._log("\n💡 双击图中红色边框的节点，可打开 SIL 验证工具。")