#!/usr/bin/env python3
"""Plot the main piecewise baseline against sweep-1 and sweep-2 schedules."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageColor, ImageDraw, ImageFont


@dataclass(frozen=True)
class RunSpec:
    """One validation run used in the comparison figure."""

    key: str
    label: str
    family: str
    config_path: Path
    metrics_path: Path
    color: str
    dash: tuple[int, int] | None
    width: int
    highlight: bool = False


@dataclass(frozen=True)
class CurveSpec:
    """One visible curve in the trajectory charts."""

    key: str
    label: str
    source_run_key: str
    color: str
    dash: tuple[int, int] | None
    width: int


@dataclass(frozen=True)
class ScheduleConfig:
    """Schedule-relevant subset of the resolved config."""

    schedule_type: str
    num_steps: int
    sigma_min: float
    sigma_max: float
    rho: float
    sampling_schedule_piecewise_power: float
    sampling_schedule_start_power: float | None
    sampling_schedule_end_power: float | None
    sampling_schedule_midpoint: float
    sampling_schedule_endpoint_trim: float
    sampling_schedule_global_u_power: float
    sampling_schedule_churn_fraction: float
    sampling_schedule_ode_fraction: float
    sampling_schedule_middle_power: float
    sampling_schedule_churn_power: float
    sampling_schedule_ode_power: float
    ode_time_duration: float
    churn_until_time: float
    churn_factor: float


@dataclass(frozen=True)
class RunResult:
    """Loaded run metrics and derived schedule stats."""

    spec: RunSpec
    cfg: ScheduleConfig
    metrics: dict[str, float]
    times: list[float]
    abs_dt: list[float]
    ode_start_step: int
    ode_start_time: float
    ode_prev_time: float


@dataclass(frozen=True)
class Panel:
    """Simple drawing panel."""

    x: int
    y: int
    width: int
    height: int


DEFAULT_RUNS = [
    RunSpec(
        key="main_piecewise",
        label="Main piecewise",
        family="Main",
        config_path=Path(
            "tmp/validation/ecsi-main-piecewise5-first200/"
            "main_piecewise5/resolved_config.yaml"
        ),
        metrics_path=Path(
            "tmp/validation/ecsi-main-piecewise5-first200/main_piecewise5/metrics.json"
        ),
        color="#1F2937",
        dash=(14, 8),
        width=6,
        highlight=True,
    ),
    RunSpec(
        key="s1_baseline",
        label="Sweep 1 baseline",
        family="Sweep 1",
        config_path=Path(
            "tmp/validation/ecsi-ode-sweep1-first200/baseline/resolved_config.yaml"
        ),
        metrics_path=Path(
            "tmp/validation/ecsi-ode-sweep1-first200/baseline/metrics.json"
        ),
        color="#93C5FD",
        dash=(8, 5),
        width=4,
    ),
    RunSpec(
        key="s1_case1",
        label="Sweep 1 case 1",
        family="Sweep 1",
        config_path=Path(
            "tmp/validation/ecsi-ode-sweep1-first200/ode050_frac030/resolved_config.yaml"
        ),
        metrics_path=Path(
            "tmp/validation/ecsi-ode-sweep1-first200/ode050_frac030/metrics.json"
        ),
        color="#3B82F6",
        dash=None,
        width=4,
    ),
    RunSpec(
        key="s1_best",
        label="Sweep 1 best",
        family="Sweep 1",
        config_path=Path(
            "tmp/validation/ecsi-ode-sweep1-first200/ode040_frac025/resolved_config.yaml"
        ),
        metrics_path=Path(
            "tmp/validation/ecsi-ode-sweep1-first200/ode040_frac025/metrics.json"
        ),
        color="#1D4ED8",
        dash=None,
        width=5,
    ),
    RunSpec(
        key="s2_baseline",
        label="Sweep 2 baseline",
        family="Sweep 2",
        config_path=Path(
            "tmp/validation/ecsi-churn-sweep2-bestode-first200/baseline/"
            "resolved_config.yaml"
        ),
        metrics_path=Path(
            "tmp/validation/ecsi-churn-sweep2-bestode-first200/baseline/metrics.json"
        ),
        color="#1D4ED8",
        dash=(3, 4),
        width=3,
    ),
    RunSpec(
        key="current_best",
        label="Sweep 2 case 1",
        family="Sweep 2",
        config_path=Path(
            "tmp/validation/ecsi-churn-sweep2-bestode-first200/case_1/"
            "resolved_config.yaml"
        ),
        metrics_path=Path(
            "tmp/validation/ecsi-churn-sweep2-bestode-first200/case_1/metrics.json"
        ),
        color="#D97706",
        dash=None,
        width=6,
        highlight=True,
    ),
    RunSpec(
        key="s2_case2",
        label="Sweep 2 case 2",
        family="Sweep 2",
        config_path=Path(
            "tmp/validation/ecsi-churn-sweep2-bestode-first200/case_2/"
            "resolved_config.yaml"
        ),
        metrics_path=Path(
            "tmp/validation/ecsi-churn-sweep2-bestode-first200/case_2/metrics.json"
        ),
        color="#059669",
        dash=None,
        width=4,
    ),
]

CURVES = [
    CurveSpec(
        key="main_piecewise",
        label="Main piecewise",
        source_run_key="main_piecewise",
        color="#1F2937",
        dash=(14, 8),
        width=6,
    ),
    CurveSpec(
        key="s1_baseline",
        label="Sweep 1 baseline",
        source_run_key="s1_baseline",
        color="#93C5FD",
        dash=(8, 5),
        width=4,
    ),
    CurveSpec(
        key="s1_case1",
        label="Sweep 1 case 1",
        source_run_key="s1_case1",
        color="#3B82F6",
        dash=None,
        width=4,
    ),
    CurveSpec(
        key="shared_best",
        label="Sweep 1 best / Sweep 2 baseline",
        source_run_key="s1_best",
        color="#1D4ED8",
        dash=None,
        width=5,
    ),
    CurveSpec(
        key="current_best",
        label="Current best (Sweep 2 case 1)",
        source_run_key="current_best",
        color="#D97706",
        dash=None,
        width=6,
    ),
    CurveSpec(
        key="s2_case2",
        label="Sweep 2 case 2",
        source_run_key="s2_case2",
        color="#059669",
        dash=None,
        width=4,
    ),
]

DELTA_CURVES = [
    "s1_baseline",
    "s1_case1",
    "shared_best",
    "current_best",
    "s2_case2",
]

METRIC_KEYS = [
    "rcsb-val/monitor/top1_lddt",
    "rcsb-val/monitor/top1_rmsd",
    "rcsb-val/monitor/top5_rmsd",
    "rcsb-val/monitor/weighted_lddt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot main-piecewise vs sweep-1 and sweep-2 schedules."
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("tmp/analysis/ecsi-main-vs-sweeps-trajectory"),
        help="Output directory for the PNG and JSON summary.",
    )
    return parser.parse_args()


def load_schedule_config(path: Path) -> ScheduleConfig:
    cfg = OmegaConf.load(path).model.structure_module
    return ScheduleConfig(
        schedule_type=str(cfg.sampling_schedule_type),
        num_steps=int(cfg.num_steps),
        sigma_min=float(cfg.sigma_min),
        sigma_max=float(cfg.sigma_max),
        rho=float(cfg.rho),
        sampling_schedule_piecewise_power=float(cfg.sampling_schedule_piecewise_power),
        sampling_schedule_start_power=(
            None
            if cfg.sampling_schedule_start_power is None
            else float(cfg.sampling_schedule_start_power)
        ),
        sampling_schedule_end_power=(
            None
            if cfg.sampling_schedule_end_power is None
            else float(cfg.sampling_schedule_end_power)
        ),
        sampling_schedule_midpoint=float(cfg.sampling_schedule_midpoint),
        sampling_schedule_endpoint_trim=float(cfg.sampling_schedule_endpoint_trim),
        sampling_schedule_global_u_power=float(cfg.sampling_schedule_global_u_power),
        sampling_schedule_churn_fraction=float(cfg.sampling_schedule_churn_fraction),
        sampling_schedule_ode_fraction=float(cfg.sampling_schedule_ode_fraction),
        sampling_schedule_middle_power=float(cfg.sampling_schedule_middle_power),
        sampling_schedule_churn_power=float(cfg.sampling_schedule_churn_power),
        sampling_schedule_ode_power=float(cfg.sampling_schedule_ode_power),
        ode_time_duration=float(cfg.ode_time_duration),
        churn_until_time=float(cfg.churn_until_time),
        churn_factor=float(cfg.churn_factor),
    )


def load_metrics(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    metrics = payload["metrics"] if "metrics" in payload else payload
    return {key: float(value) for key, value in metrics.items() if value == value}


def piecewise_power_curve(
    u_value: np.ndarray,
    start_power: float,
    end_power: float,
    midpoint: float,
) -> np.ndarray:
    t_unit = np.empty_like(u_value)
    left = u_value <= midpoint
    t_unit[left] = 1.0 - 0.5 * np.power(u_value[left] / midpoint, start_power)
    t_unit[~left] = 0.5 * np.power(
        (1.0 - u_value[~left]) / (1.0 - midpoint),
        end_power,
    )
    return np.clip(t_unit, 0.0, 1.0)


def build_piecewise_schedule(cfg: ScheduleConfig) -> np.ndarray:
    power = cfg.sampling_schedule_piecewise_power
    start_power = (
        power
        if cfg.sampling_schedule_start_power is None
        else cfg.sampling_schedule_start_power
    )
    end_power = (
        power
        if cfg.sampling_schedule_end_power is None
        else cfg.sampling_schedule_end_power
    )
    steps = np.arange(cfg.num_steps, dtype=float)
    u_value = steps / (cfg.num_steps - 1)
    if cfg.sampling_schedule_endpoint_trim > 0.0:
        endpoint_trim = cfg.sampling_schedule_endpoint_trim
        u_value = endpoint_trim + (1.0 - 2.0 * endpoint_trim) * u_value
    t_unit = piecewise_power_curve(
        u_value=u_value,
        start_power=start_power,
        end_power=end_power,
        midpoint=cfg.sampling_schedule_midpoint,
    )
    if cfg.sampling_schedule_endpoint_trim > 0.0:
        endpoint_trim = cfg.sampling_schedule_endpoint_trim
        start_u = np.array([endpoint_trim], dtype=float)
        end_u = np.array([1.0 - endpoint_trim], dtype=float)
        start_value = piecewise_power_curve(
            u_value=start_u,
            start_power=start_power,
            end_power=end_power,
            midpoint=cfg.sampling_schedule_midpoint,
        )
        end_value = piecewise_power_curve(
            u_value=end_u,
            start_power=start_power,
            end_power=end_power,
            midpoint=cfg.sampling_schedule_midpoint,
        )
        t_unit = (t_unit - end_value) / (start_value - end_value)
    return cfg.sigma_min + (cfg.sigma_max - cfg.sigma_min) * t_unit


def build_phase_power_schedule(cfg: ScheduleConfig) -> np.ndarray:
    total_scale = cfg.sigma_max - cfg.sigma_min
    churn_unit = (cfg.churn_until_time - cfg.sigma_min) / total_scale
    ode_unit = (cfg.ode_time_duration - cfg.sigma_min) / total_scale
    churn_u = churn_unit**cfg.sampling_schedule_global_u_power
    ode_u = ode_unit**cfg.sampling_schedule_global_u_power
    steps = np.arange(cfg.num_steps, dtype=float)
    s_value = steps / (cfg.num_steps - 1)
    churn_boundary = cfg.sampling_schedule_churn_fraction
    ode_boundary = 1.0 - cfg.sampling_schedule_ode_fraction
    u_value = np.empty_like(s_value)
    head_mask = s_value <= churn_boundary
    mid_mask = (s_value > churn_boundary) & (s_value <= ode_boundary)
    tail_mask = s_value > ode_boundary
    head_progress = s_value[head_mask] / churn_boundary
    mid_progress = (s_value[mid_mask] - churn_boundary) / (ode_boundary - churn_boundary)
    tail_progress = (s_value[tail_mask] - ode_boundary) / (1.0 - ode_boundary)
    u_value[head_mask] = 1.0 - (1.0 - churn_u) * np.power(
        head_progress,
        cfg.sampling_schedule_churn_power,
    )
    u_value[mid_mask] = churn_u - (churn_u - ode_u) * np.power(
        mid_progress,
        cfg.sampling_schedule_middle_power,
    )
    u_value[tail_mask] = ode_u * np.power(
        1.0 - tail_progress,
        cfg.sampling_schedule_ode_power,
    )
    t_unit = np.power(
        np.clip(u_value, 0.0, 1.0),
        1.0 / cfg.sampling_schedule_global_u_power,
    )
    return cfg.sigma_min + total_scale * t_unit


def build_schedule(cfg: ScheduleConfig) -> np.ndarray:
    if cfg.schedule_type == "piecewise_power":
        times = build_piecewise_schedule(cfg)
    elif cfg.schedule_type == "phase_power":
        times = build_phase_power_schedule(cfg)
    else:
        raise ValueError(f"Unsupported schedule type: {cfg.schedule_type}")
    return np.pad(times, (0, 1), constant_values=0.0)


def load_run(spec: RunSpec) -> RunResult:
    cfg = load_schedule_config(spec.config_path)
    metrics = load_metrics(spec.metrics_path)
    times = build_schedule(cfg)
    current = times[:-1]
    abs_dt = np.abs(np.diff(times))
    ode_indices = np.nonzero(current <= cfg.ode_time_duration)[0]
    ode_start_step = int(ode_indices[0])
    return RunResult(
        spec=spec,
        cfg=cfg,
        metrics=metrics,
        times=times.astype(float).tolist(),
        abs_dt=abs_dt.astype(float).tolist(),
        ode_start_step=ode_start_step,
        ode_start_time=float(current[ode_start_step]),
        ode_prev_time=float(current[max(ode_start_step - 1, 0)]),
    )


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def hex_rgba(color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    rgb = ImageColor.getrgb(color)
    return rgb[0], rgb[1], rgb[2], alpha


def load_font(
    size: int, bold: bool = False
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
            ]
        )
    candidates.extend(
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        ]
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    size: int,
    fill: str,
    bold: bool = False,
    anchor: str = "la",
) -> None:
    draw.text(
        xy,
        text,
        fill=fill,
        font=load_font(size, bold=bold),
        anchor=anchor,
    )


def draw_background(width: int, height: int) -> Image.Image:
    start = np.array(ImageColor.getrgb("#F7F4EE"), dtype=np.float32)
    end = np.array(ImageColor.getrgb("#EEF4FA"), dtype=np.float32)
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    blend = ((x + y) / 2.0)[..., None]
    rgb = start + (end - start) * blend
    alpha = np.full((height, width, 1), 255, dtype=np.float32)
    rgba = np.concatenate([rgb, alpha], axis=-1).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def draw_panel(
    image: Image.Image,
    panel: Panel,
    *,
    title: str,
    subtitle: str | None = None,
) -> None:
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (panel.x, panel.y, panel.x + panel.width, panel.y + panel.height),
        radius=24,
        fill=hex_rgba("#FFFFFF"),
        outline="#D6DCE5",
        width=1,
    )
    draw_text(
        draw,
        (panel.x + 22, panel.y + 18),
        title,
        size=22,
        fill="#0F172A",
        bold=True,
    )
    if subtitle is not None:
        draw_text(
            draw,
            (panel.x + 22, panel.y + 46),
            subtitle,
            size=13,
            fill="#667085",
        )


def dashed_segment_points(
    start: tuple[float, float],
    end: tuple[float, float],
    dash: tuple[int, int],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    x1, y1 = start
    x2, y2 = end
    length = float(np.hypot(x2 - x1, y2 - y1))
    if length == 0.0:
        return []
    dx = (x2 - x1) / length
    dy = (y2 - y1) / length
    pattern = [float(dash[0]), float(dash[1])]
    position = 0.0
    on = True
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    while position < length:
        segment_length = pattern[0 if on else 1]
        next_position = min(length, position + segment_length)
        if on:
            sx = x1 + dx * position
            sy = y1 + dy * position
            ex = x1 + dx * next_position
            ey = y1 + dy * next_position
            segments.append(((sx, sy), (ex, ey)))
        position = next_position
        on = not on
    return segments


def draw_polyline(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    *,
    color: str,
    width: int,
    dash: tuple[int, int] | None = None,
) -> None:
    if dash is None:
        draw.line(points, fill=color, width=width, joint="curve")
        return
    for start, end in zip(points[:-1], points[1:], strict=True):
        for seg_start, seg_end in dashed_segment_points(start, end, dash):
            draw.line((*seg_start, *seg_end), fill=color, width=width)


def plot_panel_bounds(
    panel: Panel,
    *,
    left_pad: int = 76,
    top_pad: int = 96,
    right_pad: int = 26,
    bottom_pad: int = 68,
) -> tuple[int, int, int, int]:
    left = panel.x + left_pad
    top = panel.y + top_pad
    right = panel.x + panel.width - right_pad
    bottom = panel.y + panel.height - bottom_pad
    return left, top, right, bottom


def map_point(
    x_value: float,
    y_value: float,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    bounds: tuple[int, int, int, int],
) -> tuple[float, float]:
    left, top, right, bottom = bounds
    x = left + (x_value - x_min) / (x_max - x_min) * (right - left)
    y = bottom - (y_value - y_min) / (y_max - y_min) * (bottom - top)
    return x, y


def draw_axes(
    image: Image.Image,
    panel: Panel,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    x_ticks: list[float],
    y_ticks: list[float],
    x_label: str,
    y_label: str,
    left_pad: int = 76,
    top_pad: int = 96,
    right_pad: int = 26,
    bottom_pad: int = 68,
) -> tuple[int, int, int, int]:
    draw = ImageDraw.Draw(image)
    bounds = plot_panel_bounds(
        panel,
        left_pad=left_pad,
        top_pad=top_pad,
        right_pad=right_pad,
        bottom_pad=bottom_pad,
    )
    left, top, right, bottom = bounds

    for tick in y_ticks:
        _, y = map_point(
            x_min,
            tick,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            bounds=bounds,
        )
        draw.line((left, y, right, y), fill="#E8EDF4", width=1)
        draw_text(
            draw,
            (left - 10, int(round(y))),
            fmt(tick, 3).rstrip("0").rstrip("."),
            size=12,
            fill="#667085",
            anchor="ra",
        )

    for tick in x_ticks:
        x, _ = map_point(
            tick,
            y_min,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            bounds=bounds,
        )
        draw.line((x, top, x, bottom), fill="#F0F3F8", width=1)
        draw_text(
            draw,
            (int(round(x)), bottom + 22),
            str(int(tick)),
            size=12,
            fill="#667085",
            anchor="ma",
        )

    draw.line((left, bottom, right, bottom), fill="#B8C2D3", width=2)
    draw.line((left, top, left, bottom), fill="#B8C2D3", width=2)
    draw_text(
        draw,
        ((left + right) // 2, bottom + 46),
        x_label,
        size=13,
        fill="#344054",
        anchor="ma",
    )
    draw_text(
        draw,
        (panel.x + 22, panel.y + 62),
        y_label,
        size=13,
        fill="#344054",
        bold=True,
    )
    return bounds


def draw_legend(
    image: Image.Image,
    panel: Panel,
    curves: list[CurveSpec],
) -> None:
    draw = ImageDraw.Draw(image)
    x_cursor = panel.x + 28
    y = panel.y + panel.height - 62
    row_height = 24
    max_x = panel.x + panel.width - 260
    for curve in curves:
        if x_cursor > max_x:
            x_cursor = panel.x + 28
            y += row_height
        line_y = y + 10
        if curve.dash is None:
            draw.line(
                (x_cursor, line_y, x_cursor + 36, line_y),
                fill=curve.color,
                width=curve.width,
            )
        else:
            for start, end in dashed_segment_points(
                (x_cursor, line_y),
                (x_cursor + 36, line_y),
                curve.dash,
            ):
                draw.line((*start, *end), fill=curve.color, width=curve.width)
        draw_text(
            draw,
            (x_cursor + 46, y),
            curve.label,
            size=13,
            fill="#344054",
        )
        x_cursor += 350


def trajectory_points(
    times: list[float],
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    bounds: tuple[int, int, int, int],
    clip_to_domain: bool = False,
) -> list[tuple[float, float]]:
    points = [
        map_point(
            float(step_idx),
            float(value),
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            bounds=bounds,
        )
        for step_idx, value in enumerate(times)
        if not clip_to_domain or (x_min <= float(step_idx) <= x_max)
    ]
    return points


def draw_trajectory_panel(
    image: Image.Image,
    panel: Panel,
    curves: list[CurveSpec],
    results: dict[str, RunResult],
) -> None:
    draw_panel(
        image,
        panel,
        title="Time Trajectory",
        subtitle="Main piecewise, sweep 1, sweep 2, and the current best in one view.",
    )
    bounds = draw_axes(
        image,
        panel,
        x_min=0.0,
        x_max=200.0,
        y_min=0.0,
        y_max=1.0,
        x_ticks=[0, 25, 50, 75, 100, 125, 150, 175, 200],
        y_ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        x_label="Sampling step index",
        y_label="Current time level",
        top_pad=104,
        bottom_pad=118,
    )
    draw = ImageDraw.Draw(image)

    for curve in curves:
        result = results[curve.source_run_key]
        points = trajectory_points(
            result.times,
            x_min=0.0,
            x_max=200.0,
            y_min=0.0,
            y_max=1.0,
            bounds=bounds,
        )
        draw_polyline(
            draw,
            points,
            color=curve.color,
            width=curve.width,
            dash=curve.dash,
        )

    marker_specs = [
        ("Main ODE start", results["main_piecewise"], "#1F2937"),
        ("Sweep 1 best ODE start", results["s1_best"], "#1D4ED8"),
        ("Current best ODE start", results["current_best"], "#D97706"),
    ]
    box_offsets = [0, 52, 104]
    for (label, result, color), offset in zip(marker_specs, box_offsets, strict=True):
        x, y = map_point(
            float(result.ode_start_step),
            result.ode_start_time,
            x_min=0.0,
            x_max=200.0,
            y_min=0.0,
            y_max=1.0,
            bounds=bounds,
        )
        draw.line((x, y, x, bounds[3]), fill=color, width=2)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="#FFFFFF", outline=color, width=2)
        card_x = min(int(x + 14), bounds[2] - 260)
        card_y = max(bounds[1] + 20, int(y - 20 + offset))
        draw.rounded_rectangle(
            (card_x, card_y, card_x + 246, card_y + 44),
            radius=12,
            fill=hex_rgba("#FFFFFF", 245),
            outline=color,
            width=2,
        )
        draw_text(
            draw,
            (card_x + 12, card_y + 8),
            f"{label}: step {result.ode_start_step}",
            size=12,
            fill=color,
            bold=True,
        )
        draw_text(
            draw,
            (card_x + 12, card_y + 25),
            f"t={fmt(result.ode_start_time)}",
            size=11,
            fill="#475467",
        )

    draw_legend(image, panel, curves)


def draw_zoom_panel(
    image: Image.Image,
    panel: Panel,
    curves: list[CurveSpec],
    results: dict[str, RunResult],
) -> None:
    draw_panel(
        image,
        panel,
        title="Late-Stage Zoom",
        subtitle="Focus on the tail where ODE locking and churn reallocation differ.",
    )
    bounds = draw_axes(
        image,
        panel,
        x_min=70.0,
        x_max=200.0,
        y_min=0.0,
        y_max=0.72,
        x_ticks=[80, 100, 120, 140, 160, 180, 200],
        y_ticks=[0.0, 0.2, 0.4, 0.6],
        x_label="Sampling step index",
        y_label="Current time level",
        top_pad=102,
        bottom_pad=72,
    )
    draw = ImageDraw.Draw(image)
    for curve in curves:
        result = results[curve.source_run_key]
        points = trajectory_points(
            result.times,
            x_min=70.0,
            x_max=200.0,
            y_min=0.0,
            y_max=0.72,
            bounds=bounds,
            clip_to_domain=True,
        )
        draw_polyline(
            draw,
            points,
            color=curve.color,
            width=curve.width,
            dash=curve.dash,
        )


def draw_delta_panel(
    image: Image.Image,
    panel: Panel,
    curves: list[CurveSpec],
    results: dict[str, RunResult],
) -> None:
    draw_panel(
        image,
        panel,
        title="Difference From Main",
        subtitle="Positive values mean the case stays at a higher t than main piecewise.",
    )
    bounds = draw_axes(
        image,
        panel,
        x_min=0.0,
        x_max=200.0,
        y_min=-0.25,
        y_max=0.5,
        x_ticks=[0, 25, 50, 75, 100, 125, 150, 175, 200],
        y_ticks=[-0.2, 0.0, 0.2, 0.4],
        x_label="Sampling step index",
        y_label="Delta t vs main",
        top_pad=102,
        bottom_pad=72,
    )
    draw = ImageDraw.Draw(image)
    main_times = np.array(results["main_piecewise"].times)
    for curve in curves:
        if curve.key not in DELTA_CURVES:
            continue
        times = np.array(results[curve.source_run_key].times)
        delta = (times - main_times).tolist()
        points = trajectory_points(
            delta,
            x_min=0.0,
            x_max=200.0,
            y_min=-0.25,
            y_max=0.5,
            bounds=bounds,
        )
        draw_polyline(
            draw,
            points,
            color=curve.color,
            width=max(3, curve.width - 1),
            dash=curve.dash,
        )

    zero_y = map_point(
        0.0,
        0.0,
        x_min=0.0,
        x_max=200.0,
        y_min=-0.25,
        y_max=0.5,
        bounds=bounds,
    )[1]
    draw.line((bounds[0], zero_y, bounds[2], zero_y), fill="#AAB6C8", width=2)

    main = results["main_piecewise"]
    current_best = results["current_best"]
    metrics = current_best.metrics
    base_metrics = main.metrics
    delta_top1_lddt = (
        metrics["rcsb-val/monitor/top1_lddt"] - base_metrics["rcsb-val/monitor/top1_lddt"]
    )
    delta_top1_rmsd = (
        metrics["rcsb-val/monitor/top1_rmsd"] - base_metrics["rcsb-val/monitor/top1_rmsd"]
    )
    delta_top5_rmsd = (
        metrics["rcsb-val/monitor/top5_rmsd"] - base_metrics["rcsb-val/monitor/top5_rmsd"]
    )
    delta_weighted_lddt = (
        metrics["rcsb-val/monitor/weighted_lddt"]
        - base_metrics["rcsb-val/monitor/weighted_lddt"]
    )
    delta_pp_if_top1 = (
        metrics["rcsb-val/top1/interface/lddt-protein_protein"]
        - base_metrics["rcsb-val/top1/interface/lddt-protein_protein"]
    )
    callout_x = panel.x + panel.width - 330
    callout_y = panel.y + 82
    draw.rounded_rectangle(
        (callout_x, callout_y, callout_x + 280, callout_y + 198),
        radius=16,
        fill=hex_rgba("#FFF7ED"),
        outline="#D97706",
        width=2,
    )
    draw_text(
        draw,
        (callout_x + 14, callout_y + 12),
        "Current best vs main",
        size=14,
        fill="#9A3412",
        bold=True,
    )
    lines = [
        (
            "ODE start",
            f"{main.ode_start_step} -> {current_best.ode_start_step} "
            f"(+{current_best.ode_start_step - main.ode_start_step})",
        ),
        (
            "top1_lddt",
            f"{delta_top1_lddt:+.4f}",
        ),
        (
            "top1_rmsd",
            f"{delta_top1_rmsd:+.4f}",
        ),
        (
            "top5_rmsd",
            f"{delta_top5_rmsd:+.4f}",
        ),
        (
            "weighted_lddt",
            f"{delta_weighted_lddt:+.4f}",
        ),
        (
            "pp_if_top1",
            f"{delta_pp_if_top1:+.4f}",
        ),
    ]
    for idx, (left_text, right_text) in enumerate(lines):
        y = callout_y + 42 + idx * 24
        draw_text(draw, (callout_x + 14, y), left_text, size=12, fill="#7C2D12")
        draw_text(
            draw,
            (callout_x + 266, y),
            right_text,
            size=12,
            fill="#7C2D12",
            anchor="ra",
        )


def schedule_summary_text(result: RunResult) -> str:
    if result.cfg.schedule_type == "piecewise_power":
        return (
            "piecewise "
            f"p={fmt(result.cfg.sampling_schedule_piecewise_power, 1)}, "
            f"mid={fmt(result.cfg.sampling_schedule_midpoint, 2)}"
        )
    return (
        f"ode={fmt(result.cfg.ode_time_duration, 1)}/"
        f"{fmt(result.cfg.sampling_schedule_ode_fraction, 2)}, "
        f"churn={fmt(result.cfg.churn_until_time, 2)}/"
        f"{fmt(result.cfg.sampling_schedule_churn_fraction, 2)}, "
        f"cf={fmt(result.cfg.churn_factor, 1)}"
    )


def draw_summary_panel(
    image: Image.Image,
    panel: Panel,
    results: list[RunResult],
) -> None:
    draw_panel(
        image,
        panel,
        title="Run Summary",
        subtitle=(
            "Sweep 1 best and Sweep 2 baseline share the same schedule trajectory. "
            "Their metric gap reflects rerun variance, not a different time grid."
        ),
    )
    draw = ImageDraw.Draw(image)
    table_x = panel.x + 22
    table_y = panel.y + 88
    row_height = 48
    columns = [
        ("Run", 220),
        ("Family", 110),
        ("Schedule summary", 640),
        ("ODE start", 130),
        ("top1_lddt", 130),
        ("top1_rmsd", 130),
        ("top5_rmsd", 130),
        ("weighted_lddt", 150),
    ]

    cursor_x = table_x
    for title, width in columns:
        draw.rounded_rectangle(
            (cursor_x, table_y, cursor_x + width, table_y + row_height),
            radius=8,
            fill=hex_rgba("#EEF2F8"),
            outline="#D6DCE5",
            width=1,
        )
        draw_text(
            draw,
            (cursor_x + 10, table_y + 14),
            title,
            size=14,
            fill="#344054",
            bold=True,
        )
        cursor_x += width + 6

    for row_idx, result in enumerate(results, start=1):
        y = table_y + row_idx * (row_height + 6)
        if result.spec.key == "main_piecewise":
            row_fill = "#F3F4F6"
            row_outline = "#6B7280"
        elif result.spec.key == "current_best":
            row_fill = "#FFF7ED"
            row_outline = "#D97706"
        else:
            row_fill = "#FFFFFF" if row_idx % 2 else "#FAFBFC"
            row_outline = "#E1E7F0"

        row_values = [
            result.spec.label,
            result.spec.family,
            schedule_summary_text(result),
            str(result.ode_start_step),
            fmt(result.metrics["rcsb-val/monitor/top1_lddt"], 4),
            fmt(result.metrics["rcsb-val/monitor/top1_rmsd"], 4),
            fmt(result.metrics["rcsb-val/monitor/top5_rmsd"], 4),
            fmt(result.metrics["rcsb-val/monitor/weighted_lddt"], 4),
        ]

        cursor_x = table_x
        for value, (_, width) in zip(row_values, columns, strict=True):
            draw.rounded_rectangle(
                (cursor_x, y, cursor_x + width, y + row_height),
                radius=8,
                fill=hex_rgba(row_fill),
                outline=row_outline,
                width=1,
            )
            draw_text(
                draw,
                (cursor_x + 10, y + 14),
                value,
                size=13,
                fill="#111827",
                bold=cursor_x == table_x,
            )
            cursor_x += width + 6


def render_png(
    out_path: Path,
    results: dict[str, RunResult],
) -> None:
    width = 2400
    height = 1940
    image = draw_background(width, height)
    draw = ImageDraw.Draw(image)

    draw_text(
        draw,
        (70, 56),
        "Main Baseline, Sweep 1, Sweep 2, and the Current Best",
        size=34,
        fill="#0F172A",
        bold=True,
    )
    draw_text(
        draw,
        (70, 92),
        (
            "Trajectory view: previous main branch piecewise vs phase-power "
            "follow-up sweeps. Metrics use the first-200 validation subset."
        ),
        size=17,
        fill="#475467",
    )

    trajectory_panel = Panel(70, 130, 2260, 780)
    zoom_panel = Panel(70, 940, 1100, 420)
    delta_panel = Panel(1230, 940, 1100, 420)
    summary_panel = Panel(70, 1390, 2260, 500)

    curve_lookup = {curve.key: curve for curve in CURVES}
    ordered_curves = [curve_lookup[key] for key in [curve.key for curve in CURVES]]
    ordered_results = [results[spec.key] for spec in DEFAULT_RUNS]

    draw_trajectory_panel(image, trajectory_panel, ordered_curves, results)
    draw_zoom_panel(image, zoom_panel, ordered_curves, results)
    draw_delta_panel(image, delta_panel, ordered_curves, results)
    draw_summary_panel(image, summary_panel, ordered_results)

    image.save(out_path)


def build_json_summary(results: dict[str, RunResult]) -> dict[str, Any]:
    main = results["main_piecewise"]
    current_best = results["current_best"]
    compare = {
        "main_vs_current_best": {
            "ode_start_step_main": main.ode_start_step,
            "ode_start_step_current_best": current_best.ode_start_step,
            "ode_start_delay_steps": current_best.ode_start_step - main.ode_start_step,
            "top1_lddt_delta": current_best.metrics["rcsb-val/monitor/top1_lddt"]
            - main.metrics["rcsb-val/monitor/top1_lddt"],
            "top1_rmsd_delta": current_best.metrics["rcsb-val/monitor/top1_rmsd"]
            - main.metrics["rcsb-val/monitor/top1_rmsd"],
            "top5_rmsd_delta": current_best.metrics["rcsb-val/monitor/top5_rmsd"]
            - main.metrics["rcsb-val/monitor/top5_rmsd"],
            "weighted_lddt_delta": current_best.metrics["rcsb-val/monitor/weighted_lddt"]
            - main.metrics["rcsb-val/monitor/weighted_lddt"],
            "pp_interface_top1_delta": current_best.metrics[
                "rcsb-val/top1/interface/lddt-protein_protein"
            ]
            - main.metrics["rcsb-val/top1/interface/lddt-protein_protein"],
        }
    }
    return {
        "runs": {
            key: {
                "label": result.spec.label,
                "family": result.spec.family,
                "config_path": str(result.spec.config_path.resolve()),
                "metrics_path": str(result.spec.metrics_path.resolve()),
                "schedule_type": result.cfg.schedule_type,
                "ode_start_step": result.ode_start_step,
                "ode_start_time": result.ode_start_time,
                "metrics": {metric: result.metrics[metric] for metric in METRIC_KEYS},
            }
            for key, result in results.items()
        },
        "comparison": compare,
    }


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {spec.key: load_run(spec) for spec in DEFAULT_RUNS}
    png_path = out_dir / "ecsi_main_vs_sweeps_trajectory.png"
    json_path = out_dir / "ecsi_main_vs_sweeps_summary.json"

    render_png(png_path, results)
    json_path.write_text(
        json.dumps(build_json_summary(results), indent=2),
        encoding="utf-8",
    )

    print(f"PNG saved to: {png_path}")
    print(f"JSON saved to: {json_path}")
    print(
        "Main piecewise vs current best: "
        f"ODE start {results['main_piecewise'].ode_start_step} -> "
        f"{results['current_best'].ode_start_step}"
    )


if __name__ == "__main__":
    main()
