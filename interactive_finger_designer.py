# finger_designer.py

import sys
import numpy as np
from scipy.spatial import ConvexHull

import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QGroupBox, QLabel, QDoubleSpinBox, QScrollArea, QSplitter,
    QFrame, QSizePolicy, QSlider, QPushButton
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont

from dataclasses import dataclass, asdict, field
import json

from BRANCH_human_finger_dev.ValeroCuervasModel import pose_dict

from strucMatrices import VariableStrucMatrix, secondaryDev
from finger import Finger, valeroFinger, valero, dof_constraints
from utils import hull_containment_scalar

import argparse

import traceback

import time

import warnings

# ─── Configuration ────────────────────────────────────────────────────────────

# POSES_TO_SHOW = {
#     'Extended':     [10 * np.pi / 180] * 3,
#     'Intermediate': [45 * np.pi / 180, 10 * np.pi / 180, 10 * np.pi / 180],
#     'Flexed':       [45 * np.pi / 180, 45 * np.pi / 180, 10 * np.pi / 180],
# }

POSES_TO_SHOW = ['Extension', 'Intermediate', 'Flexion']

R               = secondaryDev.R
D               = secondaryDev.D

# How long to wait after last slider move before recomputing (ms)
# Prevents thrashing during rapid drags
DEBOUNCE_MS = 50

# Colors matching your existing plotting style
COLOR_DESIGN   = pg.mkColor('#00BFFF')   # cyan-ish, matches color='cyan'
COLOR_DESIGN_PF   = pg.mkColor("#FF00FF")   # magenta-ish
COLOR_VALERO   = pg.mkColor('#FF6B6B')   # red
COLOR_FEASIBLE = pg.mkColor('#44FF44')   # green constraint indicator
COLOR_VIOLATED = pg.mkColor('#FF4444')   # red constraint indicator
COLOR_WARN     = pg.mkColor('#FFB347')   # orange — close to limit

numFlexs  = 3
numExts   = 3
numJoints = 3

# ─── Joint parameter model ────────────────────────────────────────────────────

@dataclass
class JointParams:
    joint_idx: int
    e_max: float = 0.25
    e_min: float = 0.125
    f_max: float = 0.42
    f_min: float = 0.00
    f_min_inner: float = 0.20
    length: float = 1.4

def load_joint_params(path: str) -> list[JointParams]:
    with open(path) as f:
        return [JointParams(**d) for d in json.load(f)]

# class JointParams:
#     """All editable parameters for one joint."""
#     def __init__(self, joint_idx: int,
#                  e_max=0.25, e_min=0.125,
#                  f_max=0.42, f_min=0.0, f_min_inner=0.20,
#                  length=1.4):
#         self.joint_idx    = joint_idx
#         # Extension tendons
#         self.e_max        = e_max
#         self.e_min        = e_min
#         # Fexion tendons
#         self.f_max        = f_max
#         self.f_min        = f_min
#         self.f_min_inner  = f_min_inner
#         self.length       = length

    # def to_routing_entry(self):
    #     if self.is_extension:
    #         return (self.max, self.min)   # (max, min) for extensor
    #     else:
    #         return (0, self.max, self.min) # (0, max, min) for flexor


# ─── Finger builder ───────────────────────────────────────────────────────────

def build_finger(joint_params: list[JointParams]) -> Finger:
    """
    Construct a Finger from the current GUI state.
    Mirrors createFingerFromVector but driven by JointParams directly.
    """
    # flexors    = [jp for jp in joint_params if not jp.is_extension]
    # extensors  = [jp for jp in joint_params if jp.is_extension]
    lengths    = [jp.length for jp in joint_params]

    fs = [(jp.f_min, jp.f_max, jp.f_min_inner) for jp in joint_params]
    es = [(jp.e_min, jp.e_max)    for jp in joint_params]

    ranges = []
    ranges += [val for i in range(numJoints) for val in [es[i]] + [fs[i]] * (numJoints - i)]

    types = []
    types += [VariableStrucMatrix.convergent_circles_extension_joint if x==-1
         else VariableStrucMatrix.ideal_convergent_circles_joint_with_limit
                                    for x in D[np.where(D!=0)]]

    VSM = VariableStrucMatrix(
        R, D, F=[50] * (numJoints + 1),
        ranges=ranges,
        types=types,
        name="DesignerFinger"
    )
    return Finger(VSM, lengths[:numJoints], tensionLimit=TENSION_LIMIT)


# ─── Capability computation ───────────────────────────────────────────────────

def get_hull_vertices_2d(finger: Finger, pose: list, force_pf=False) -> np.ndarray | None:
    """
    Returns (N, 2) array of hull vertex coordinates for plotting,
    or None if the hull can't be computed.
    """
    constraints = dof_constraints[finger.numJoints]['task_dof']  \
                                   if (ENFORCE_POINT_FORCES or force_pf) else []

    try:
        hull = finger.get_convex_capability(
            pose,
            constrained_dof=constraints,
            output_dim=2,
            space='task'
        )
        pts = hull.points
        if pts.shape[1] > 2:
            pts = pts[:, :2]
            hull = ConvexHull(pts)
        # Return vertices in order for a closed polygon
        verts = hull.points[hull.vertices]
        # Close the polygon
        return np.vstack([verts, verts[0]])
    except Exception:
        return None

def get_valero_hull_vertices_2d(pose: list) -> np.ndarray | None:
    try:
        hull = valeroFinger.get_convex_capability(
            pose,
            f_max=valero.Fo(pose),
            constrained_dof=dof_constraints[valeroFinger.numJoints]['task_dof'],
            space='task'
        )
        pts = hull.points
        if pts.shape[1] > 2:
            pts = pts[:, :2]
            hull = ConvexHull(pts)
        verts = hull.points[hull.vertices]
        return np.vstack([verts, verts[0]])
    except Exception:
        print("[VALERO] We have not returned a valero hull")
        traceback.print_exc()
        return None

def compute_constraints(finger: Finger,
                        design_hulls:  list[ConvexHull] | None,
                        valero_hulls:  list[ConvexHull] | None,
                        overall_thickness: float) -> dict:
    """
    Evaluate all constraints and return a dict of
    {name: (value, limit, satisfied, description)}.
    """
    results = {}

    #region Bend radius
    # print(f"  [constraints] finger={finger}, effortFunctions={getattr(finger.structure, 'effortFunctions', 'MISSING')}")
    try:
        rs = [fn.r for fn in finger.structure.effortFunctions]
        min_r = np.min(rs)
        min_r_idx = np.argmin(rs)
        jtp_of_interest = finger.structure.effortFunctions[min_r_idx].idx
        #TODO: add r1 for flexion joints
        results['Bend Radius'] = (
            min_r, MIN_BEND_RADIUS,
            min_r >= MIN_BEND_RADIUS,
            f"min r = {min_r:.3f} at jtp {tuple(int(i) for i in jtp_of_interest)} (limit ≥ {MIN_BEND_RADIUS})"
        )
    except Exception as e:
        traceback.print_exc()
        results['Bend Radius'] = (None, MIN_BEND_RADIUS, False, str(e))

    #endregion

    #region Envelope containment
    # print(f"  [constraints] design_hull={design_hull}, valero_hull={valero_hull}")
    # print(len(design_hulls), len(valero_hulls))
    if design_hulls is not None and valero_hulls is not None:
        assert len(design_hulls)==len(pose_dict)
        assert len(valero_hulls)==len(pose_dict)
        containments=[]
        for design_hull, valero_hull in zip(design_hulls, valero_hulls):
            try:
                containment = hull_containment_scalar(valero_hull, design_hull)
                # containment = hull_containment_scalar(design_hull, valero_hull)
                containments.append(containment)
                # print("tried successfully")
            except Exception as e:
                # print("entered exception")
                err = e
                traceback.print_exc()
                continue
        # print(len(design_hulls), len(valero_hulls))
        # print(len(containments))
        if len(containments)==0:
            results['Envelopes Valero'] = (None, 0.0, False,
                                                  f"All hulls failed")
        else:
            limit_containment = np.max(containments)
            results['Envelopes Valero'] = (
                limit_containment, 0.0,
                limit_containment <= 0.0,
                f"containment = {limit_containment:.4f}  (want ≤ 0)"
            )

    #endregion

    #region Overall thickness (just display the computed value)
    # print(f"  [constraints] overall_thickness={overall_thickness}")
    try:
        results['Thickness'] = (
            overall_thickness, 0.65,
            True,
            f"thickness = {overall_thickness:.4f}"
        )
    except Exception as e:
        print(f"  [constraints] THICKNESS FAILED: {e}")
        traceback.print_exc()
        results['Thickness'] = (None, 0.65, False, str(e))
    #endregion

    #region Fits in Length constraint
    exceedances=[]
    exceeded=[]

    for i in range(len(finger.lengths)):
        length_limit = finger.lengths[i] if i==len(finger.lengths)-1 \
                                                        else finger.lengths[i]/2
        exceedance = (length_limit-
             finger.structure.indexed_effort_functions[i,i+1].c*np.cos(np.pi/4))
        exceedances.append(exceedance)
        if exceedance<0:
            exceeded.append(i)
    exceedances=np.array(exceedances)

    if np.all(exceedances>0):
        results['Length Limit'] = (None, None, True,
                            "All designed surfaces fit within designed lengths")

    else:
        results['Length Limit'] = (np.max(exceedances), None, False,
          f"Joint(s) {exceeded} have exceed by {-exceedances[exceeded]}")
    #endregion

    # print(f"  [constraints] returning: {list(results.keys())}")

    return results


# ─── Plot helpers ─────────────────────────────────────────────────────────────

def _add_origin_marker(pw: pg.PlotWidget):
    """Crosshair marker at the origin."""
    pw.addItem(pg.InfiniteLine(pos=0, angle=90,
                               pen=pg.mkPen('#ffffff', width=1,
                                            style=Qt.PenStyle.DashLine)))
    pw.addItem(pg.InfiniteLine(pos=0, angle=0,
                               pen=pg.mkPen('#ffffff', width=1,
                                            style=Qt.PenStyle.DashLine)))

# ─── Parameter spinbox widget ─────────────────────────────────────────────────

class ParamSpinBox(QWidget):
    """Label + spinbox pair."""
    def __init__(self, label: str, value: float,
                 lo: float, hi: float,
                 step: float = 0.005,
                 callback=None):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        lbl = QLabel(label)
        lbl.setMinimumWidth(130)
        lbl.setFont(QFont("Monospace", 9))
        layout.addWidget(lbl)

        self.spin = QDoubleSpinBox()
        self.spin.setRange(lo, hi)
        self.spin.setValue(value)
        self.spin.setSingleStep(step)
        self.spin.setDecimals(4)
        self.spin.setMinimumWidth(90)
        layout.addWidget(self.spin)

        self.value_label = lbl
        if callback:
            self.spin.valueChanged.connect(callback)

    def value(self) -> float:
        return self.spin.value()

    def set_value(self, v: float):
        self.spin.blockSignals(True)
        self.spin.setValue(v)
        self.spin.blockSignals(False)


# Range Slider

class RangeSlider(QWidget):
    """
    Horizontal slider with two draggable handles (min, max)
    plus spinboxes for manual entry.
    """
    changed = pg.QtCore.Signal()   # emits when either handle moves

    def __init__(self, label: str,
                 lo: float, hi: float,
                 val_min: float, val_max: float,
                 step: float = 0.005):
        super().__init__()
        self._lo   = lo
        self._hi   = hi
        self._step = step
        self._dragging = None   # 'min' | 'max' | None

        layout = QVBoxLayout(self)
        layout.setSpacing(2)
        layout.setContentsMargins(0, 2, 0, 2)

        # ── Label ─────────────────────────────────────────────────────────────
        lbl = QLabel(label)
        lbl.setFont(QFont("Monospace", 9))
        layout.addWidget(lbl)

        # ── Spinboxes ─────────────────────────────────────────────────────────
        spin_row = QWidget()
        spin_layout = QHBoxLayout(spin_row)
        spin_layout.setContentsMargins(0, 0, 0, 0)

        self._spin_min = QDoubleSpinBox()
        self._spin_max = QDoubleSpinBox()
        for sp, val in ((self._spin_min, val_min),
                        (self._spin_max, val_max)):
            sp.blockSignals(True)
            sp.setDecimals(4)
            sp.setSingleStep(step)
            sp.setRange(lo, hi)
            sp.setValue(val)
            sp.setMinimumWidth(80)
            sp.blockSignals(False)
            spin_layout.addWidget(sp)

        layout.addWidget(spin_row)

        # ── Custom slider canvas ───────────────────────────────────────────────
        self._canvas = _RangeSliderCanvas(lo, hi, val_min, val_max)
        self._canvas.setFixedHeight(28)
        self._canvas.handle_moved.connect(self._on_handle_moved)
        layout.addWidget(self._canvas)

        # Wire spinboxes → canvas
        self._spin_min.valueChanged.connect(self._on_spin_min)
        self._spin_max.valueChanged.connect(self._on_spin_max)

    # ── Public API ────────────────────────────────────────────────────────────

    def min_value(self) -> float:
        return self._spin_min.value()

    def max_value(self) -> float:
        return self._spin_max.value()

    def set_values(self, val_min: float, val_max: float):
        self._spin_min.blockSignals(True)
        self._spin_max.blockSignals(True)
        self._spin_min.setValue(val_min)
        self._spin_max.setValue(val_max)
        self._spin_min.blockSignals(False)
        self._spin_max.blockSignals(False)
        self._canvas.set_values(val_min, val_max)

    # ── Internal slots ────────────────────────────────────────────────────────

    def _on_handle_moved(self, val_min: float, val_max: float):
        self._spin_min.blockSignals(True)
        self._spin_max.blockSignals(True)
        self._spin_min.setValue(val_min)
        self._spin_max.setValue(val_max)
        self._spin_min.blockSignals(False)
        self._spin_max.blockSignals(False)
        self.changed.emit()

    def _on_spin_min(self, v: float):
        # Clamp so min never exceeds max
        if v > self._spin_max.value():
            self._spin_min.blockSignals(True)
            self._spin_min.setValue(self._spin_max.value())
            self._spin_min.blockSignals(False)
            v = self._spin_max.value()
        self._canvas.set_values(v, self._spin_max.value())
        self.changed.emit()

    def _on_spin_max(self, v: float):
        if v < self._spin_min.value():
            self._spin_max.blockSignals(True)
            self._spin_max.setValue(self._spin_min.value())
            self._spin_max.blockSignals(False)
            v = self._spin_min.value()
        self._canvas.set_values(self._spin_min.value(), v)
        self.changed.emit()


class _RangeSliderCanvas(QWidget):
    """
    The actual painted slider track + two draggable handles.
    Separated from RangeSlider so painting logic is self-contained.
    """
    handle_moved = pg.QtCore.Signal(float, float)  # (val_min, val_max)

    HANDLE_R  = 7    # handle circle radius px
    TRACK_H   = 4    # track bar height px

    def __init__(self, lo: float, hi: float,
                 val_min: float, val_max: float):
        super().__init__()
        self._lo      = lo
        self._hi      = hi
        self._val_min = val_min
        self._val_max = val_max
        self._drag    = None   # 'min' | 'max'
        self.setMouseTracking(True)

    # ── Public ────────────────────────────────────────────────────────────────

    def set_values(self, val_min: float, val_max: float):
        self._val_min = val_min
        self._val_max = val_max
        self.update()   # triggers paintEvent

    # ── Coordinate helpers ────────────────────────────────────────────────────

    def _val_to_x(self, v: float) -> int:
        r = self.HANDLE_R
        w = self.width() - 2 * r
        return int(r + (v - self._lo) / (self._hi - self._lo) * w)

    def _x_to_val(self, x: int) -> float:
        r = self.HANDLE_R
        w = self.width() - 2 * r
        v = self._lo + (x - r) / w * (self._hi - self._lo)
        return float(np.clip(v, self._lo, self._hi))

    def _handle_hit(self, x: int, y: int) -> str | None:
        cy = self.height() // 2
        for name, v in (('min', self._val_min), ('max', self._val_max)):
            hx = self._val_to_x(v)
            if (x - hx) ** 2 + (y - cy) ** 2 <= (self.HANDLE_R + 3) ** 2:
                return name
        return None

    # ── Paint ─────────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        from PyQt6.QtGui import QPainter, QBrush, QPen, QColor
        p   = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w   = self.width()
        cy  = self.height() // 2
        r   = self.HANDLE_R
        th  = self.TRACK_H

        # Track background
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor('#444444')))
        p.drawRoundedRect(r, cy - th // 2, w - 2 * r, th, 2, 2)

        # Active range (between handles)
        x_min = self._val_to_x(self._val_min)
        x_max = self._val_to_x(self._val_max)
        p.setBrush(QBrush(QColor('#2d7fc1')))
        p.drawRoundedRect(x_min, cy - th // 2,
                          x_max - x_min, th, 2, 2)

        # Handles
        for val, color in ((self._val_min, '#00BFFF'),
                           (self._val_max, '#FF6B6B')):
            hx = self._val_to_x(val)
            p.setBrush(QBrush(QColor(color)))
            p.setPen(QPen(QColor('#ffffff'), 1.5))
            p.drawEllipse(hx - r, cy - r, 2 * r, 2 * r)

        p.end()

    # ── Mouse ─────────────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        self._drag = self._handle_hit(event.position().x(),
                                      event.position().y())

    def mouseMoveEvent(self, event):
        if self._drag is None:
            # Change cursor when hovering over a handle
            hit = self._handle_hit(event.position().x(),
                                   event.position().y())
            self.setCursor(Qt.CursorShape.SizeHorCursor if hit
                           else Qt.CursorShape.ArrowCursor)
            return

        v = self._x_to_val(int(event.position().x()))
        if self._drag == 'min':
            self._val_min = min(v, self._val_max)
        else:
            self._val_max = max(v, self._val_min)
        self.update()
        self.handle_moved.emit(self._val_min, self._val_max)

    def mouseReleaseEvent(self, event):
        self._drag = None
        self.setCursor(Qt.CursorShape.ArrowCursor)

# ─── Joint panel ──────────────────────────────────────────────────────────────

class JointPanel(QGroupBox):
    def __init__(self, joint_params: JointParams, on_changed):
        label = f"Joint {joint_params.joint_idx + 1}"
        super().__init__(label)
        self._params     = joint_params
        self._on_changed = on_changed
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(2)

        # Extension range slider
        self._e_range = RangeSlider(
            "ext:  min  ◄────►  max",
            lo=0.125, hi=0.5,
            val_min=self._params.e_min,
            val_max=self._params.e_max,
        )
        self._e_range.changed.connect(self._changed)
        layout.addWidget(self._e_range)

        # Flexion range slider — note f_min is 0 (outer), f_min_inner is inner
        self._f_range = RangeSlider(
            "flx:  min  ◄────►  max",
            lo=0.0, hi=0.5,
            val_min=self._params.f_min,
            val_max=self._params.f_max,
        )
        self._f_range.changed.connect(self._changed)
        layout.addWidget(self._f_range)

        # f_min (outer) — stays as a spinbox since it's usually fixed at 0
        self._f_min_inner = ParamSpinBox(
            "flx: f_min (outer)", self._params.f_min_inner,
            0.0, 0.5, step=0.005,
            callback=self._changed
        )
        layout.addWidget(self._f_min_inner)

        self._length = ParamSpinBox(
            "length (in)", self._params.length,
            0.5, 3.0, step=0.01,
            callback=self._changed
        )
        layout.addWidget(self._length)

        self._thick_label = QLabel("thickness: —")
        self._thick_label.setFont(QFont("Monospace", 8))
        layout.addWidget(self._thick_label)

    def _changed(self):
        self._sync_to_params()
        self._on_changed()

    def _sync_to_params(self):
        self._params.e_min       = self._e_range.min_value()
        self._params.e_max       = self._e_range.max_value()
        self._params.f_min       = self._f_range.min_value()
        self._params.f_max       = self._f_range.max_value()
        self._params.f_min_inner = self._f_min_inner.value()
        self._params.length      = self._length.value()

    def update_thickness_display(self, thickness: float | None = None):
        if thickness is not None:
            self._thick_label.setText(f"thickness contribution: {thickness:.4f}")
        else:
            self._thick_label.setText(
                f"ext [{self._params.e_min:.3f}, {self._params.e_max:.3f}]  "
                f"flx [{self._params.f_min:.3f}, {self._params.f_max:.3f}]"
            )

# ─── Constraint indicator ─────────────────────────────────────────────────────

class ConstraintIndicator(QWidget):
    """Traffic-light style constraint display."""
    def __init__(self, name: str):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        self._dot = QLabel("●")
        self._dot.setFont(QFont("Monospace", 14))
        self._dot.setFixedWidth(24)
        layout.addWidget(self._dot)

        self._label = QLabel(name)
        self._label.setFont(QFont("Monospace", 9))
        layout.addWidget(self._label)

        self._detail = QLabel("")
        self._detail.setFont(QFont("Monospace", 8))
        self._detail.setStyleSheet("color: #888888;")
        layout.addWidget(self._detail)

        layout.addStretch()
        self.set_state(None, "—")

    def set_state(self, satisfied: bool | None, detail: str):
        if satisfied is None:
            self._dot.setStyleSheet("color: #666666;")
        elif satisfied:
            self._dot.setStyleSheet(f"color: {COLOR_FEASIBLE.name()};")
        else:
            self._dot.setStyleSheet(f"color: {COLOR_VIOLATED.name()};")
        self._detail.setText(detail)


# ─── Main window ──────────────────────────────────────────────────────────────

class FingerDesigner(QMainWindow):
    def __init__(self, default_params=None):
        super().__init__()
        self.setWindowTitle("Interactive Finger Designer")
        self.resize(1600, 900)

        # Debounce timer — recompute only after user stops moving sliders
        self._debounce = QTimer()
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self._recompute)
        self._recomputing = False

        # Pre-compute Valero hulls (done once at startup)
        self._valero_hulls = []
        self._valero_verts = {}
        for pose_name in POSES_TO_SHOW:
            hull  = valeroFinger.get_convex_capability(
                            pose_dict[pose_name],
                            f_max=valero.Fo(pose_dict[pose_name]),
                            constrained_dof=
                            dof_constraints[valeroFinger.numJoints]['task_dof'],
                            space='task'
                        )
            verts = get_valero_hull_vertices_2d(pose_dict[pose_name])
            self._valero_verts[pose_name] = verts
            self._valero_hulls.append(hull)

        # Initial joint parameters

        if default_params is None:
            self._default_joint_params = [
                # Default moment arms, custom lengths
                JointParams(0, e_min=0.165, e_max=0.3,
                            f_max = 0.4, f_min=0.12, f_min_inner=0.2,
                            length=1.4),
                JointParams(1, e_min=0.13, e_max=0.3,
                            f_max = 0.38, f_min=0.1, f_min_inner=0.2,
                            length=1.4),
                JointParams(2, e_min=0.13, e_max=0.3,
                            f_max = 0.34, f_min=0.1, f_min_inner=0.2,
                            length=1),
            ]
        else:
            self._default_joint_params = default_params
        self._joint_params = self._default_joint_params

        self._build_ui()

        self._current_finger = None
        self._pose_explorer  = None
        # self._pose_explorer = PoseExplorerWindow(lambda: self._current_finger)
        # self._pose_explorer.show()

        self._recompute()   # Initial render

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(splitter)

        # Left: controls
        splitter.addWidget(self._build_control_panel())
        splitter.setStretchFactor(0, 0)

        # Right: plots
        splitter.addWidget(self._build_plot_panel())
        splitter.setStretchFactor(1, 1)

        splitter.setSizes([380, 1200])

    def _build_control_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(420)      # slightly wider for the extra slider
        layout = QVBoxLayout(panel)

        # Global readout
        global_box    = QGroupBox("Structure Properties")
        global_layout = QVBoxLayout(global_box)
        self._thickness_label = QLabel("Overall Thickness: —")
        self._thickness_label.setFont(QFont("Monospace", 10))
        global_layout.addWidget(self._thickness_label)

        # ── Pose input for S matrix display ──────────────────────────────────────
        global_layout.addWidget(QLabel("S matrix at pose:"))

        pose_row = QWidget()
        pose_row_layout = QHBoxLayout(pose_row)
        pose_row_layout.setContentsMargins(0, 0, 0, 0)

        self._s_pose_spinboxes = []
        for i in range(numJoints):
            sb = QDoubleSpinBox()
            sb.setRange(0.0, np.pi / 2)
            sb.setValue(0.0)
            sb.setSingleStep(0.01)
            sb.setDecimals(3)
            sb.setPrefix(f"q{i+1}: ")
            sb.valueChanged.connect(self._on_param_changed)   # reuse debounce
            pose_row_layout.addWidget(sb)
            self._s_pose_spinboxes.append(sb)

        global_layout.addWidget(pose_row)

        # S matrix display — monospace, fixed-width, enough room for all entries
        self._s_matrix_label = QLabel("S: —")
        self._s_matrix_label.setFont(QFont("Monospace", 8))
        self._s_matrix_label.setWordWrap(True)
        # Enough width for e.g. a 4×4 matrix at 4dp: "0.1234  " × 4 cols × 4 rows
        self._s_matrix_label.setMinimumHeight(80)
        self._s_matrix_label.setStyleSheet(
            "background-color: #1e1e1e; padding: 4px; border: 1px solid #444;"
        )
        self._s_matrix_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse   # allow copy-paste
        )
        global_layout.addWidget(self._s_matrix_label)

        layout.addWidget(global_box)

        # Constraints
        con_box    = QGroupBox("Constraints")
        con_layout = QVBoxLayout(con_box)
        self._constraint_indicators = {}
        for name in ['Bend Radius', 'Envelopes Valero', 'Thickness', 'Length Limit']:
            ind = ConstraintIndicator(name)
            self._constraint_indicators[name] = ind
            con_layout.addWidget(ind)
        layout.addWidget(con_box)

        # One panel per joint — no more ext/flx headers
        scroll         = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout  = QVBoxLayout(scroll_content)

        self._joint_panels: list[JointPanel]
        self._joint_panels = []
        for jp in self._joint_params:
            panel_w = JointPanel(jp, self._on_param_changed)
            self._joint_panels.append(panel_w)
            scroll_layout.addWidget(panel_w)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        actions_box = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_box)

        reset_btn = QPushButton("Reset to Default")
        reset_btn.clicked.connect(self._reset_to_defaults)
        actions_layout.addWidget(reset_btn)

        save_btn = QPushButton("Export Design")
        save_btn.clicked.connect(self._dump_joint_params)
        actions_layout.addWidget(save_btn)

        pose_explorer_btn = QPushButton("Launch Pose Explorer")
        pose_explorer_btn.clicked.connect(self._launch_pose_explorer)
        actions_layout.addWidget(pose_explorer_btn)

        layout.addWidget(actions_box)

        return panel

    def _launch_pose_explorer(self):
        if self._pose_explorer is None:
            self._pose_explorer = PoseExplorerWindow(lambda: self._current_finger)
        self._pose_explorer.show()
        self._pose_explorer.raise_()

    def _dump_joint_params(self):
        json_string = [asdict(jp) for jp in self._joint_params]
        json_path = f"finger_designs/design_{numJoints}dof_{"pf" if ENFORCE_POINT_FORCES else "npf"}_{time.strftime('%Y%m%d-%H%M')}.json"
        with open(json_path, 'w') as f:
            json.dump(json_string, f, indent=2)
        print(f"Saved {json_path}")

    def _build_plot_panel(self) -> QWidget:
        """
        One pyqtgraph PlotWidget per pose, arranged in a row.
        """
        panel  = QWidget()
        layout = QHBoxLayout(panel)
        layout.setSpacing(4)

        self._plot_widgets  = {}   # pose_name → PlotWidget
        self._design_curves = {}   # pose_name → PlotDataItem  (filled polygon)
        if not ENFORCE_POINT_FORCES:
            self._pf_curves = {}
        self._valero_curves = {}   # pose_name → PlotDataItem

        for pose_name in POSES_TO_SHOW:
            # Container with title
            col = QWidget()
            col_layout = QVBoxLayout(col)
            col_layout.setSpacing(2)

            title = QLabel(pose_name)
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title.setFont(QFont("Monospace", 10, QFont.Weight.Bold))
            col_layout.addWidget(title)

            pw = pg.PlotWidget()
            pw.setAspectLocked(True)
            pw.showGrid(x=True, y=True, alpha=0.3)
            pw.setLabel('left',   'Palmar (lbf)')
            pw.setLabel('bottom', 'Distal (lbf)')
            pw.addLegend(offset=(10, 10))
            _add_origin_marker(pw)

            # Valero hull — filled, drawn first so design is on top
            valero_curve = pw.plot(
                [], [],
                name='Valero',
                pen=pg.mkPen(COLOR_VALERO, width=2),
                fillLevel=0,
                brush=pg.mkBrush(
                    QColor(COLOR_VALERO.red(),
                           COLOR_VALERO.green(),
                           COLOR_VALERO.blue(), 50)   # translucent fill
                ),
            )

            if not ENFORCE_POINT_FORCES:
                # If we aren't enforcing point forces for the purpose of
                # constraints, plot that version anyway just so we can see it
                design_curve_pf = pw.plot(
                    [], [],
                    name='Design with Point Forces',
                    pen = pg.mkPen(COLOR_DESIGN_PF, width=2),
                    fillLevel=0,
                    brush=pg.mkBrush(
                        QColor(COLOR_DESIGN_PF.red(),
                               COLOR_DESIGN_PF.green(),
                               COLOR_DESIGN_PF.blue(), 70)
                    ),
                )

            # Design hull — filled
            design_curve = pw.plot(
                [], [],
                name='Design',
                pen=pg.mkPen(COLOR_DESIGN, width=2),
                fillLevel=0,
                brush=pg.mkBrush(
                    QColor(COLOR_DESIGN.red(),
                           COLOR_DESIGN.green(),
                           COLOR_DESIGN.blue(), 60)
                ),
            )

            self._plot_widgets[pose_name]  = pw
            self._design_curves[pose_name] = design_curve
            if not ENFORCE_POINT_FORCES:
                self._pf_curves[pose_name] = design_curve_pf
            self._valero_curves[pose_name] = valero_curve

            col_layout.addWidget(pw)
            layout.addWidget(col)

            # Draw Valero hull immediately — it never changes
            verts = self._valero_verts.get(pose_name)
            if verts is not None:
                valero_curve.setData(verts[:, 0], verts[:, 1])

        return panel

    # ── Parameter change handling ─────────────────────────────────────────────

    def _on_param_changed(self):
        """Called by any spinbox change. Starts/restarts the debounce timer."""
        self._debounce.start(DEBOUNCE_MS)

    # ── Core recompute ────────────────────────────────────────────────────────

    def _recompute(self):
        """
        Rebuild the finger from current params and update all plots +
        constraint indicators. Called after debounce fires.
        """

        if self._recomputing:
            self._debounce.start(DEBOUNCE_MS)
            return
        self._recomputing=True
        # print("recompute called")
        try:
            # ── Build finger ──────────────────────────────────────────────────────
            # t0 = time.perf_counter()
            try:
                finger = build_finger(self._joint_params)
                # update current finger to send to other window
                self._current_finger = finger
            except Exception as e:
                self._set_all_constraints_error(str(e))
                self._clear_design_curves()
                return

            # ── Compute overall thickness ─────────────────────────────────────────

            thicknesses = [
                jp.f_max + jp.e_min
                for jp in self._joint_params
            ]
            thicknesses += [
                jp.e_max + jp.f_min + np.sqrt(2)/2*
        (finger.structure.indexed_effort_functions[jp.joint_idx, jp.joint_idx+1].c -
        finger.structure.indexed_effort_functions[jp.joint_idx, jp.joint_idx+1].r)
                for jp in self._joint_params
            ]
            # print(len(thicknesses))
            overall_thicknesses = np.vstack((np.array(thicknesses[:numJoints]),
                                             np.array(thicknesses[numJoints:])))
            overall_thicknesses = np.max(overall_thicknesses, axis=0)
            overall_thickness = max(overall_thicknesses)

            # Update thickness readout
            self._thickness_label.setText(
                f"Overall thickness: {overall_thickness:.4f} in"
            )

            # Update per-panel display
            # for i, (panel_w, jp) in enumerate(zip(self._joint_panels, self._joint_params)):
            #     panel_w.update_thickness_display(thicknesses[i])
            for i, panel_w in enumerate(self._joint_panels):
                panel_w.update_thickness_display(overall_thicknesses[i])

            # ── S matrix display ──────────────────────────────────────────────────────
            try:
                s_pose = [sb.value() for sb in self._s_pose_spinboxes]
                S = finger.structure.S(s_pose)   # or however your structure matrix is called
                # Format each row: 4dp, space-separated, right-aligned
                rows = []
                for row in S:
                    rows.append("  " + "  ".join(f"{v:8.4f}" for v in row))
                self._s_matrix_label.setText("S =\n" + "\n".join(rows))
            except Exception as e:
                self._s_matrix_label.setText(f"S: err ({e})")

            # ── Update plots for each pose ────────────────────────────────────────

            design_hulls = []
            for i, pose_name in enumerate(POSES_TO_SHOW):
                # TODO: Make this more robust later
                pose = pose_dict[pose_name]
                design_pose = pose[1:] if numJoints == 3 else pose
                # print(len(pose))
                try:
                    design_hull = finger.get_convex_capability(
                        design_pose,
                        constrained_dof=
                            dof_constraints[finger.numJoints]['task_dof']
                            if ENFORCE_POINT_FORCES else [],
                        output_dim=2,
                        space='task'
                    )
                    # Extract verts from the hull we already computed
                    pts   = design_hull.points
                    verts = pts[design_hull.vertices]
                    verts = np.vstack([verts, verts[0]])   # close polygon
                    self._design_curves[pose_name].setData(verts[:, 0], verts[:, 1])
                    design_hulls.append(design_hull)
                except Exception as e:
                    print(f"  hull failed [{pose_name}]: {e}")
                    self._design_curves[pose_name].setData([], [])

                if not ENFORCE_POINT_FORCES:
                    try:
                        pf_hull = finger.get_convex_capability(
                            design_pose,
                            constrained_dof=
                                dof_constraints[finger.numJoints]['task_dof'],
                            output_dim=2,
                            space='task'
                        )
                        pf_pts   = pf_hull.points
                        pf_verts = pf_pts[pf_hull.vertices]
                        pf_verts = np.vstack([pf_verts, pf_verts[0]])
                        self._pf_curves[pose_name].setData(pf_verts[:, 0], pf_verts[:, 1])
                    except Exception:
                        self._pf_curves[pose_name].setData([], [])

                # Refit axes to data
                # self._plot_widgets[pose_name].enableAutoRange()

            # print(f"  [recompute] calling compute_constraints with "
            #   f"finger={finger}, "
            #   f"design_hull={first_design_hull}, "
            #   f"valero_hull={first_valero_hull}, "
            #   f"thickness={overall_thickness}")
            # ── Evaluate and display constraints ──────────────────────────────────
            con_results = compute_constraints(
                finger,
                design_hulls,
                self._valero_hulls,
                overall_thickness
            )

            for name, (value, limit, satisfied, detail) in con_results.items():
                if name in self._constraint_indicators:
                    self._constraint_indicators[name].set_state(satisfied, detail)

            if self._pose_explorer is not None:
                self._pose_explorer.refresh()
            # print(f"  [recompute] {(time.perf_counter()-t0)*1000:.1f}ms")
        finally:
            self._recomputing = False

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _reset_to_defaults(self):
        import copy
        self._joint_params = copy.deepcopy(self._default_joint_params)
        for panel_w, jp in zip(self._joint_panels, self._joint_params):
            panel_w.params = jp
            panel_w._e_range.set_values(jp.e_min, jp.e_max)
            panel_w._f_range.set_values(jp.f_min, jp.f_max)
            panel_w._f_min_inner.set_value(jp.f_min_inner)
            panel_w._length.set_value(jp.length)
        self._on_param_changed()

    def _clear_design_curves(self):
        for curve in self._design_curves.values():
            curve.setData([], [])

    def _set_all_constraints_error(self, msg: str):
        for ind in self._constraint_indicators.values():
            ind.set_state(False, f"Error: {msg[:60]}")

# ─── Pose Explorer window ─────────────────────────────────────────────────────

class PoseExplorerWindow(QMainWindow):
    def __init__(self, get_finger_fn):
        super().__init__()
        self.setWindowTitle("Pose Explorer")
        self.resize(700, 600)
        self._get_finger = get_finger_fn
        self._use_degrees = False

        self._debounce = QTimer()
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self._recompute)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # ── Left: controls ────────────────────────────────────────────────────
        ctrl = QWidget()
        ctrl.setFixedWidth(300)
        ctrl_layout = QVBoxLayout(ctrl)

        # Degrees/radians toggle
        toggle_row = QWidget()
        toggle_layout = QHBoxLayout(toggle_row)
        toggle_layout.setContentsMargins(0, 0, 0, 0)
        toggle_layout.addWidget(QLabel("Units:"))
        self._unit_btn = QPushButton("Switch to Degrees")
        self._unit_btn.setCheckable(True)
        self._unit_btn.clicked.connect(self._toggle_units)
        toggle_layout.addWidget(self._unit_btn)
        toggle_layout.addStretch()
        ctrl_layout.addWidget(toggle_row)

        # One row per joint: label | slider | spinbox
        self._sliders  = []
        self._spinboxes = []

        for i in range(numJoints):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 2, 0, 2)

            lbl = QLabel(f"q{i+1}")
            lbl.setFont(QFont("Monospace", 9))
            lbl.setFixedWidth(28)
            row_layout.addWidget(lbl)

            sl = QSlider(Qt.Orientation.Horizontal)
            sl.setRange(0, 1000)      # always internal units: 0–1000 → 0–π/2
            sl.setValue(0)
            sl.setTickInterval(250)
            sl.setTickPosition(QSlider.TickPosition.TicksBelow)
            row_layout.addWidget(sl)

            sb = QDoubleSpinBox()
            sb.setDecimals(4)
            sb.setMinimumWidth(80)
            sb.setSingleStep(0.01)
            # Initial state: radians
            sb.setRange(0.0, np.pi / 2)
            sb.setValue(0.0)
            row_layout.addWidget(sb)

            # Slider → spinbox
            def _on_slide(v, sb=sb):
                rad = v / 1000 * np.pi / 2
                sb.blockSignals(True)
                sb.setValue(np.degrees(rad) if self._use_degrees else rad)
                sb.blockSignals(False)
                self._debounce.start(DEBOUNCE_MS)

            # Spinbox → slider
            def _on_spin(v, sl=sl):
                rad = np.radians(v) if self._use_degrees else v
                sl.blockSignals(True)
                sl.setValue(int(rad / (np.pi / 2) * 1000))
                sl.blockSignals(False)
                self._debounce.start(DEBOUNCE_MS)

            sl.valueChanged.connect(_on_slide)
            sb.valueChanged.connect(_on_spin)

            self._sliders.append(sl)
            self._spinboxes.append(sb)
            ctrl_layout.addWidget(row)

        ctrl_layout.addStretch()
        layout.addWidget(ctrl)

        # ── Right: plot ───────────────────────────────────────────────────────
        pw = pg.PlotWidget()
        pw.setAspectLocked(True)
        pw.showGrid(x=True, y=True, alpha=0.3)
        pw.setLabel('left',   'Palmar (lbf)')
        pw.setLabel('bottom', 'Distal (lbf)')
        pw.addLegend(offset=(10, 10))
        # Set a fixed range that won't change — adjust to your expected capability scale
        pw.setXRange(-100, 100, padding=0)
        pw.setYRange(-100, 100, padding=0)
        # pw.setMouseEnabled(x=False, y=False)   # optional: prevent user panning too
        self._plot_widget = pw
        _add_origin_marker(pw)

        self._curve = pw.plot(
            [], [], name='Design @ pose',
            pen=pg.mkPen(COLOR_DESIGN, width=2),
            fillLevel=0,
            brush=pg.mkBrush(QColor(
                COLOR_DESIGN.red(), COLOR_DESIGN.green(),
                COLOR_DESIGN.blue(), 60)),
        )
        # self._pose_label = pg.TextItem("", color='#ffffff', anchor=(0, 1))
        # pw.addItem(self._pose_label)
        self._plot_widget = pw
        layout.addWidget(pw)

    def _toggle_units(self, checked: bool):
        self._use_degrees = checked
        self._unit_btn.setText(
            "Switch to Radians" if checked else "Switch to Degrees"
        )
        # Update all spinbox ranges and current values without re-triggering
        for sl, sb in zip(self._sliders, self._spinboxes):
            rad = sl.value() / 1000 * np.pi / 2
            sb.blockSignals(True)
            if self._use_degrees:
                sb.setRange(0.0, 90.0)
                sb.setSingleStep(1.0)
                sb.setDecimals(2)
                sb.setValue(np.degrees(rad))
            else:
                sb.setRange(0.0, np.pi / 2)
                sb.setSingleStep(0.01)
                sb.setDecimals(4)
                sb.setValue(rad)
            sb.blockSignals(False)

    def _get_pose(self) -> list[float]:
        return [sl.value() / 1000 * np.pi / 2 for sl in self._sliders]

    def _recompute(self):
        pose   = self._get_pose()
        finger = self._get_finger()
        if finger is None:
            return
        verts = get_hull_vertices_2d(finger, pose)
        if verts is not None:
            self._curve.setData(verts[:, 0], verts[:, 1])
        else:
            self._curve.setData([], [])
        # self._plot_widget.enableAutoRange()
        # unit_str = "°" if self._use_degrees else "rad"
        # vals = [np.degrees(q) if self._use_degrees else q
        #         for q in pose]
        # self._pose_label.setText(
        #     "q = [" + ", ".join(f"{q:.2f}{unit_str}" for q in vals) + "]"
        # )

    def refresh(self):
        self._debounce.start(0)

# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Interactive Finger Designer")
    parser.add_argument('-pf', '--enforce-point-forces', action='store_true',
                        default=False)
    parser.add_argument('-t', '--tension-limit', type=float, default=50.0)
    parser.add_argument('--min-bend-radius', type=float, default=0.472)
    parser.add_argument('-f', '--input-file', type=str, default="default")
    return parser.parse_known_args()

# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    warnings.filterwarnings("ignore")

    args, _ = parse_args()
    global ENFORCE_POINT_FORCES, TENSION_LIMIT, MIN_BEND_RADIUS
    ENFORCE_POINT_FORCES = args.enforce_point_forces
    TENSION_LIMIT        = args.tension_limit
    MIN_BEND_RADIUS      = args.min_bend_radius
    if ENFORCE_POINT_FORCES:
        print ("Enforcing point forces on designed finger")
    if args.input_file != "default":
        print(f"Loading default design vector from {args.input_file}")
        default_parameters = load_joint_params(args.input_file)
    print(f"Tension Limit: {TENSION_LIMIT}lbf. Min Bend Radius: {MIN_BEND_RADIUS}in.")
    # Use a PURPLE :) theme throughout
    pg.setConfigOption('background', '#280137')
    pg.setConfigOption('foreground', '#dddddd')

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Dark palette
    from PyQt6.QtGui import QPalette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window,          QColor('#280137'))
    palette.setColor(QPalette.ColorRole.WindowText,      QColor('#dddddd'))
    palette.setColor(QPalette.ColorRole.Base,            QColor("#320046"))
    palette.setColor(QPalette.ColorRole.AlternateBase,   QColor("#460E5C"))
    palette.setColor(QPalette.ColorRole.Text,            QColor('#dddddd'))
    palette.setColor(QPalette.ColorRole.Button,          QColor('#3c3c3c'))
    palette.setColor(QPalette.ColorRole.ButtonText,      QColor('#dddddd'))
    palette.setColor(QPalette.ColorRole.Highlight,       QColor('#2d7fc1'))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor('#ffffff'))
    app.setPalette(palette)

    window = FingerDesigner(default_params=default_parameters)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()