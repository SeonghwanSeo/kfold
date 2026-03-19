#!/usr/bin/env python3
"""Render a static SVG comparison of ECSI phase-power time schedules.

This script mirrors the `phase_power` schedule implementation in
`src/kfold/model/modules/structure_module/kfold_ecsi.py` without importing the
full model stack. It is intended for cheap, local schedule inspection when the
question is whether the deterministic ODE phase starts too early.
"""

from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageColor, ImageDraw, ImageFont

DEFAULT_CASES = [
    "baseline:ode_time_duration=0.6,sampling_schedule_ode_fraction=0.40",
    "case_1:ode_time_duration=0.5,sampling_schedule_ode_fraction=0.30",
    "case_2:ode_time_duration=0.4,sampling_schedule_ode_fraction=0.25",
]

CASE_STYLES = [
    {
        "name": "baseline",
        "label": "Baseline",
        "color": "#355C7D",
        "accent": "#27435C",
        "soft": "#DCE6F0",
    },
    {
        "name": "case_1",
        "label": "Case 1",
        "color": "#E07A2D",
        "accent": "#B25E1F",
        "soft": "#FCE7D6",
    },
    {
        "name": "case_2",
        "label": "Case 2",
        "color": "#2A9D8F",
        "accent": "#1F786D",
        "soft": "#D8F1EC",
    },
]

PHASE_COLORS = {
    "churn": "#D9E1ED",
    "middle": "#E5EEDB",
    "ode": "#F9D8B4",
}


@dataclass(frozen=True)
class ScheduleConfig:
    """Minimal phase-power schedule contract."""

    num_steps: int
    sigma_min: float
    sigma_max: float
    sampling_schedule_type: str
    sampling_schedule_global_u_power: float
    sampling_schedule_churn_fraction: float
    sampling_schedule_churn_power: float
    sampling_schedule_middle_power: float
    sampling_schedule_ode_fraction: float
    sampling_schedule_ode_power: float
    churn_until_time: float
    ode_time_duration: float
    churn_factor: float

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> ScheduleConfig:
        return cls(
            num_steps=int(mapping["num_steps"]),
            sigma_min=float(mapping["sigma_min"]),
            sigma_max=float(mapping["sigma_max"]),
            sampling_schedule_type=str(mapping["sampling_schedule_type"]),
            sampling_schedule_global_u_power=float(
                mapping["sampling_schedule_global_u_power"]
            ),
            sampling_schedule_churn_fraction=float(
                mapping["sampling_schedule_churn_fraction"]
            ),
            sampling_schedule_churn_power=float(mapping["sampling_schedule_churn_power"]),
            sampling_schedule_middle_power=float(
                mapping["sampling_schedule_middle_power"]
            ),
            sampling_schedule_ode_fraction=float(
                mapping["sampling_schedule_ode_fraction"]
            ),
            sampling_schedule_ode_power=float(mapping["sampling_schedule_ode_power"]),
            churn_until_time=float(mapping["churn_until_time"]),
            ode_time_duration=float(mapping["ode_time_duration"]),
            churn_factor=float(mapping["churn_factor"]),
        )

    def validate(self) -> None:
        if self.sampling_schedule_type.lower() != "phase_power":
            raise ValueError(
                "This analysis script only supports sampling_schedule_type='phase_power'."
            )
        if self.num_steps <= 1:
            raise ValueError("num_steps must be > 1.")
        if self.sigma_max <= self.sigma_min:
            raise ValueError("sigma_max must be > sigma_min.")
        if self.sampling_schedule_global_u_power <= 0.0:
            raise ValueError("sampling_schedule_global_u_power must be > 0.")
        if self.sampling_schedule_churn_fraction <= 0.0:
            raise ValueError("sampling_schedule_churn_fraction must be > 0.")
        if self.sampling_schedule_ode_fraction <= 0.0:
            raise ValueError("sampling_schedule_ode_fraction must be > 0.")
        if (
            self.sampling_schedule_churn_fraction + self.sampling_schedule_ode_fraction
            >= 1.0
        ):
            raise ValueError(
                "sampling_schedule_churn_fraction + "
                "sampling_schedule_ode_fraction must be < 1."
            )
        if self.sampling_schedule_churn_power <= 1.0:
            raise ValueError("sampling_schedule_churn_power must be > 1.")
        if self.sampling_schedule_middle_power <= 0.0:
            raise ValueError("sampling_schedule_middle_power must be > 0.")
        if self.sampling_schedule_ode_power <= self.sampling_schedule_global_u_power:
            raise ValueError(
                "sampling_schedule_ode_power must exceed "
                "sampling_schedule_global_u_power."
            )
        if not (
            self.sigma_min
            < self.ode_time_duration
            < self.churn_until_time
            < self.sigma_max
        ):
            raise ValueError(
                "Expected sigma_min < ode_time_duration < churn_until_time < sigma_max."
            )


@dataclass(frozen=True)
class CaseMetrics:
    """Derived schedule metrics for one comparison case."""

    name: str
    display_name: str
    color: str
    accent: str
    soft: str
    ode_time_duration: float
    ode_fraction: float
    churn_until_time: float
    churn_fraction: float
    churn_factor: float
    churn_steps: int
    middle_steps: int
    ode_steps: int
    ode_start_step: int
    ode_start_step_fraction: float
    ode_start_time: float
    ode_prev_time: float
    mean_abs_dt_total: float
    mean_abs_dt_pre_ode: float
    mean_abs_dt_ode: float
    max_abs_dt: float
    min_abs_dt: float
    steps_per_unit_pre_ode: float
    steps_per_unit_ode: float
    times: list[float]
    abs_dt: list[float]


@dataclass(frozen=True)
class Panel:
    """Simple panel geometry in SVG coordinates."""

    x: float
    y: float
    width: float
    height: float


SVG_NS = {"svg": "http://www.w3.org/2000/svg"}

FIELD_LABELS = {
    "ode_time_duration": "ode_time_duration",
    "sampling_schedule_ode_fraction": "ode_fraction",
    "churn_until_time": "churn_until_time",
    "sampling_schedule_churn_fraction": "churn_fraction",
    "churn_factor": "churn_factor",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot ECSI phase-power time-step comparisons as a static SVG."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model/module/structure_module/ecsi.yaml"),
        help="Base structure-module config path.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("tmp/analysis/ecsi-time-step-comparison"),
        help="Directory for SVG and summary files.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=None,
        help="Optional override for the number of schedule steps.",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help=(
            "Case specification in the form "
            "'name:key=value,key2=value2'. If omitted, the default three-way "
            "comparison is used."
        ),
    )
    parser.add_argument(
        "--title",
        type=str,
        default="ECSI phase-power time schedule comparison",
        help="Figure title.",
    )
    parser.add_argument(
        "--png_scale",
        type=int,
        default=2,
        help="Rasterization scale used before downsampling the PNG.",
    )
    return parser.parse_args()


def parse_scalar(value: str) -> Any:
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_case_spec(spec: str) -> tuple[str, dict[str, Any]]:
    try:
        name, payload = spec.split(":", maxsplit=1)
    except ValueError as exc:
        raise ValueError(
            f"Invalid case spec {spec!r}. Expected 'name:key=value,...'."
        ) from exc

    overrides: dict[str, Any] = {}
    for item in payload.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            key, raw_value = item.split("=", maxsplit=1)
        except ValueError as exc:
            raise ValueError(
                f"Invalid override {item!r} in case {name!r}. Expected key=value."
            ) from exc
        overrides[key.strip()] = parse_scalar(raw_value.strip())

    if not overrides:
        raise ValueError(f"Case {name!r} did not contain any overrides.")
    return name.strip(), overrides


def load_base_config(path: Path, num_steps: int | None) -> ScheduleConfig:
    cfg = OmegaConf.load(path)
    mapping = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(mapping, dict):
        raise ValueError(f"Expected a mapping config at {path}, got {type(mapping)!r}.")
    if num_steps is not None:
        mapping["num_steps"] = num_steps
    return ScheduleConfig.from_mapping(mapping)


def apply_overrides(cfg: ScheduleConfig, overrides: dict[str, Any]) -> ScheduleConfig:
    current = asdict(cfg)
    for key, value in overrides.items():
        if key not in current:
            raise KeyError(f"Unknown schedule override key: {key}")
        current[key] = value
    updated = replace(cfg, **current)
    updated.validate()
    return updated


def style_for_case(name: str) -> dict[str, str]:
    for style in CASE_STYLES:
        if style["name"] == name:
            return style
    index = sum(ord(char) for char in name) % len(CASE_STYLES)
    base = CASE_STYLES[index]
    return {
        "name": name,
        "label": name.replace("_", " ").title(),
        "color": base["color"],
        "accent": base["accent"],
        "soft": base["soft"],
    }


def build_phase_power_schedule(cfg: ScheduleConfig) -> np.ndarray:
    """Mirror `_get_phase_power_schedule` and `get_sampling_schedule`."""

    cfg.validate()
    total_scale = cfg.sigma_max - cfg.sigma_min
    churn_unit = (cfg.churn_until_time - cfg.sigma_min) / total_scale
    ode_unit = (cfg.ode_time_duration - cfg.sigma_min) / total_scale

    churn_u = churn_unit**cfg.sampling_schedule_global_u_power
    ode_u = ode_unit**cfg.sampling_schedule_global_u_power
    if not 0.0 < ode_u < churn_u < 1.0:
        raise ValueError("Phase-power schedule produced invalid u-space bounds.")

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
    times = cfg.sigma_min + total_scale * t_unit
    return np.pad(times, (0, 1), constant_values=0.0)


def summarize_case(
    name: str,
    cfg: ScheduleConfig,
    times: np.ndarray,
) -> CaseMetrics:
    style = style_for_case(name)
    current = times[:-1]
    abs_dt = np.abs(np.diff(times))

    step_axis = np.arange(cfg.num_steps, dtype=float) / (cfg.num_steps - 1)
    churn_mask = step_axis <= cfg.sampling_schedule_churn_fraction
    ode_mask = current <= cfg.ode_time_duration
    middle_mask = (~churn_mask) & (~ode_mask)

    ode_indices = np.nonzero(ode_mask)[0]
    if len(ode_indices) == 0:
        raise ValueError(f"Case {name!r} did not enter the ODE phase.")

    ode_start = int(ode_indices[0])
    ode_prev_index = max(ode_start - 1, 0)
    pre_ode_mask = ~ode_mask

    return CaseMetrics(
        name=name,
        display_name=style["label"],
        color=style["color"],
        accent=style["accent"],
        soft=style["soft"],
        ode_time_duration=cfg.ode_time_duration,
        ode_fraction=cfg.sampling_schedule_ode_fraction,
        churn_until_time=cfg.churn_until_time,
        churn_fraction=cfg.sampling_schedule_churn_fraction,
        churn_factor=cfg.churn_factor,
        churn_steps=int(churn_mask.sum()),
        middle_steps=int(middle_mask.sum()),
        ode_steps=int(ode_mask.sum()),
        ode_start_step=ode_start,
        ode_start_step_fraction=ode_start / cfg.num_steps,
        ode_start_time=float(current[ode_start]),
        ode_prev_time=float(current[ode_prev_index]),
        mean_abs_dt_total=float(abs_dt.mean()),
        mean_abs_dt_pre_ode=float(abs_dt[pre_ode_mask].mean()),
        mean_abs_dt_ode=float(abs_dt[ode_mask].mean()),
        max_abs_dt=float(abs_dt.max()),
        min_abs_dt=float(abs_dt.min()),
        steps_per_unit_pre_ode=float(
            pre_ode_mask.sum() / (cfg.sigma_max - cfg.ode_time_duration)
        ),
        steps_per_unit_ode=float(ode_mask.sum() / cfg.ode_time_duration),
        times=times.astype(float).tolist(),
        abs_dt=abs_dt.astype(float).tolist(),
    )


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def varied_fields(configs: list[ScheduleConfig]) -> list[str]:
    candidate_fields = [
        "ode_time_duration",
        "sampling_schedule_ode_fraction",
        "churn_until_time",
        "sampling_schedule_churn_fraction",
        "churn_factor",
    ]
    fields: list[str] = []
    for field in candidate_fields:
        values = [getattr(cfg, field) for cfg in configs]
        first = values[0]
        if any(abs(value - first) > 1e-8 for value in values[1:]):
            fields.append(field)
    return fields


def format_case_value(field: str, value: float) -> str:
    if field in {
        "sampling_schedule_ode_fraction",
        "sampling_schedule_churn_fraction",
        "churn_until_time",
    }:
        return fmt(value, 2)
    if field in {"ode_time_duration", "churn_factor"}:
        return fmt(value, 1)
    return fmt(value, 3)


def case_setting_summary(cfg: ScheduleConfig, fields: list[str]) -> str:
    summary_fields = fields or [
        "ode_time_duration",
        "sampling_schedule_ode_fraction",
    ]
    return ", ".join(
        f"{FIELD_LABELS[field]}={format_case_value(field, getattr(cfg, field))}"
        for field in summary_fields
    )


def svg_text(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 16,
    weight: int = 400,
    fill: str = "#1F2937",
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="DejaVu Sans, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}">'
        f"{escape(text)}</text>"
    )


def svg_rect(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str,
    stroke: str = "none",
    stroke_width: float = 0.0,
    rx: float = 0.0,
    opacity: float = 1.0,
) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" '
        f'height="{height:.1f}" rx="{rx:.1f}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{stroke_width:.1f}" '
        f'opacity="{opacity:.3f}" />'
    )


def svg_line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str,
    stroke_width: float,
    opacity: float = 1.0,
    dash: str | None = None,
) -> str:
    dash_attr = "" if dash is None else f' stroke-dasharray="{dash}"'
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{stroke_width:.1f}" '
        f'opacity="{opacity:.3f}"{dash_attr} />'
    )


def svg_circle(
    cx: float,
    cy: float,
    radius: float,
    *,
    fill: str,
    stroke: str,
    stroke_width: float,
) -> str:
    return (
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{stroke_width:.1f}" />'
    )


def svg_polyline(
    points: list[tuple[float, float]],
    *,
    stroke: str,
    stroke_width: float,
    fill: str = "none",
    opacity: float = 1.0,
) -> str:
    point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return (
        f'<polyline points="{point_text}" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="{stroke_width:.1f}" opacity="{opacity:.3f}" '
        'stroke-linecap="round" stroke-linejoin="round" />'
    )


def panel_coords(
    panel: Panel,
    x_value: float,
    x_min: float,
    x_max: float,
    y_value: float,
    y_min: float,
    y_max: float,
) -> tuple[float, float]:
    x_span = x_max - x_min
    y_span = y_max - y_min
    x = panel.x + (x_value - x_min) / x_span * panel.width
    y = panel.y + panel.height - (y_value - y_min) / y_span * panel.height
    return x, y


def draw_panel_frame(parts: list[str], panel: Panel, title: str) -> None:
    parts.append(
        svg_rect(
            panel.x,
            panel.y,
            panel.width,
            panel.height,
            fill="#FFFFFF",
            stroke="#D6DCE5",
            stroke_width=1.0,
            rx=22.0,
        )
    )
    parts.append(
        svg_text(
            panel.x + 22.0,
            panel.y + 30.0,
            title,
            size=18,
            weight=700,
            fill="#0F172A",
        )
    )


def draw_axes(
    parts: list[str],
    panel: Panel,
    *,
    x_ticks: list[float],
    y_ticks: list[float],
    x_domain: tuple[float, float],
    y_domain: tuple[float, float],
    x_label: str,
    y_label: str,
    plot_top_padding: float = 40.0,
    plot_bottom_padding: float = 42.0,
    plot_left_padding: float = 68.0,
    plot_right_padding: float = 24.0,
) -> Panel:
    plot = Panel(
        x=panel.x + plot_left_padding,
        y=panel.y + plot_top_padding,
        width=panel.width - plot_left_padding - plot_right_padding,
        height=panel.height - plot_top_padding - plot_bottom_padding,
    )

    x_min, x_max = x_domain
    y_min, y_max = y_domain

    for tick in y_ticks:
        _, y = panel_coords(plot, x_min, x_min, x_max, tick, y_min, y_max)
        parts.append(
            svg_line(
                plot.x,
                y,
                plot.x + plot.width,
                y,
                stroke="#E8EDF4",
                stroke_width=1.0,
            )
        )
        parts.append(
            svg_text(
                plot.x - 12.0,
                y + 5.0,
                fmt(tick, 3).rstrip("0").rstrip("."),
                size=12,
                fill="#667085",
                anchor="end",
            )
        )

    for tick in x_ticks:
        x, _ = panel_coords(plot, tick, x_min, x_max, y_min, y_min, y_max)
        parts.append(
            svg_line(
                x,
                plot.y,
                x,
                plot.y + plot.height,
                stroke="#F0F3F8",
                stroke_width=1.0,
            )
        )
        parts.append(
            svg_text(
                x,
                plot.y + plot.height + 24.0,
                fmt(tick, 0),
                size=12,
                fill="#667085",
                anchor="middle",
            )
        )

    parts.append(
        svg_line(
            plot.x,
            plot.y + plot.height,
            plot.x + plot.width,
            plot.y + plot.height,
            stroke="#B8C2D3",
            stroke_width=1.2,
        )
    )
    parts.append(
        svg_line(
            plot.x,
            plot.y,
            plot.x,
            plot.y + plot.height,
            stroke="#B8C2D3",
            stroke_width=1.2,
        )
    )

    parts.append(
        svg_text(
            plot.x + plot.width / 2.0,
            panel.y + panel.height - 10.0,
            x_label,
            size=13,
            fill="#344054",
            anchor="middle",
        )
    )
    parts.append(
        svg_text(
            panel.x + 22.0,
            panel.y + 52.0,
            y_label,
            size=13,
            weight=600,
            fill="#344054",
        )
    )
    return plot


def phase_panel_caption(configs: list[ScheduleConfig], fields: list[str]) -> str:
    churn_varies = "sampling_schedule_churn_fraction" in fields
    ode_varies = "sampling_schedule_ode_fraction" in fields
    if churn_varies and not ode_varies:
        return (
            "Churn allocation varies by case; fixed ODE tail: "
            f"ode_fraction = {fmt(configs[0].sampling_schedule_ode_fraction, 2)}"
        )
    if not churn_varies:
        return (
            "Common fixed phase: churn_fraction = "
            f"{fmt(configs[0].sampling_schedule_churn_fraction, 2)}"
        )
    return "Phase allocation varies across cases."


def summary_copy(
    results: list[CaseMetrics],
    configs: list[ScheduleConfig],
    fields: list[str],
) -> tuple[str, str, str | None]:
    varied = set(fields)
    if varied and varied.issubset(
        {"ode_time_duration", "sampling_schedule_ode_fraction"}
    ):
        ode_steps_text = " to ".join(str(result.ode_start_step) for result in results)
        headline = (
            "Shrinking the ODE tail delays deterministic lock from "
            f"step {ode_steps_text}, while the fixed churn cutoff remains "
            f"at step {results[0].churn_steps}."
        )
        subline = (
            f"Fixed knobs: num_steps={configs[0].num_steps}, sigma in "
            f"[{fmt(configs[0].sigma_min)}, {fmt(configs[0].sigma_max)}], "
            f"global_u_power={fmt(configs[0].sampling_schedule_global_u_power, 1)}, "
            f"churn_until_time={fmt(configs[0].churn_until_time, 1)}."
        )
        return headline, subline, None

    if varied and varied.issubset(
        {
            "churn_until_time",
            "sampling_schedule_churn_fraction",
            "churn_factor",
        }
    ):
        churn_steps_text = " to ".join(str(result.churn_steps) for result in results)
        if len({result.ode_start_step for result in results}) == 1:
            headline = (
                "Extending the churn window moves the churn cutoff from "
                f"step {churn_steps_text}, while ODE start stays fixed at "
                f"step {results[0].ode_start_step}."
            )
        else:
            ode_steps_text = " to ".join(str(result.ode_start_step) for result in results)
            headline = (
                "Extending the churn window moves the churn cutoff from "
                f"step {churn_steps_text} and pushes ODE start from "
                f"step {ode_steps_text}."
            )
        subline = (
            "Fixed ODE tail: "
            f"ode_time_duration={fmt(configs[0].ode_time_duration, 1)}, "
            f"ode_fraction={fmt(configs[0].sampling_schedule_ode_fraction, 2)}, "
            f"num_steps={configs[0].num_steps}."
        )
        note = (
            "Note: churn_factor is included from the Obsidian candidate note, "
            "but it does not change the t_i schedule by itself."
        )
        return headline, subline, note

    field_text = ", ".join(FIELD_LABELS[field] for field in fields) or "selected knobs"
    ode_steps_text = ", ".join(str(result.ode_start_step) for result in results)
    headline = (
        f"Schedule comparison across cases with varying {field_text}. "
        f"ODE start steps are {ode_steps_text}."
    )
    subline = (
        f"Fixed base config: num_steps={configs[0].num_steps}, sigma in "
        f"[{fmt(configs[0].sigma_min)}, {fmt(configs[0].sigma_max)}], "
        f"global_u_power={fmt(configs[0].sampling_schedule_global_u_power, 1)}."
    )
    return headline, subline, None


def render_schedule_panel(
    parts: list[str],
    panel: Panel,
    results: list[CaseMetrics],
    num_steps: int,
) -> None:
    draw_panel_frame(parts, panel, "Time trajectory t_i")
    plot = draw_axes(
        parts,
        panel,
        x_ticks=[0, 50, 100, 150, 200],
        y_ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        x_domain=(0.0, float(num_steps)),
        y_domain=(0.0, 1.0),
        x_label="Sampling step index",
        y_label="Current time level",
    )

    churn_marker_x, _ = panel_coords(
        plot,
        results[0].churn_steps,
        0.0,
        float(num_steps),
        0.0,
        0.0,
        1.0,
    )
    parts.append(
        svg_line(
            churn_marker_x,
            plot.y,
            churn_marker_x,
            plot.y + plot.height,
            stroke="#AAB6C8",
            stroke_width=1.2,
            dash="6 6",
        )
    )
    parts.append(
        svg_text(
            churn_marker_x + 8.0,
            plot.y + 18.0,
            "fixed churn cutoff",
            size=12,
            fill="#667085",
        )
    )

    for result in results:
        points: list[tuple[float, float]] = []
        for step_idx, value in enumerate(result.times):
            x, y = panel_coords(
                plot,
                float(step_idx),
                0.0,
                float(num_steps),
                float(value),
                0.0,
                1.0,
            )
            points.append((x, y))
        parts.append(
            svg_polyline(
                points,
                stroke=result.color,
                stroke_width=4.0,
            )
        )

        step_x, step_y = panel_coords(
            plot,
            float(result.ode_start_step),
            0.0,
            float(num_steps),
            result.ode_start_time,
            0.0,
            1.0,
        )
        parts.append(
            svg_line(
                step_x,
                step_y,
                step_x,
                plot.y + plot.height,
                stroke=result.color,
                stroke_width=1.6,
                dash="7 7",
                opacity=0.7,
            )
        )
        parts.append(
            svg_circle(
                step_x,
                step_y,
                5.5,
                fill="#FFFFFF",
                stroke=result.color,
                stroke_width=2.4,
            )
        )

    label_offsets = [0.0, 56.0, 112.0]
    for result, offset in zip(results, label_offsets, strict=False):
        step_x, step_y = panel_coords(
            plot,
            float(result.ode_start_step),
            0.0,
            float(num_steps),
            result.ode_start_time,
            0.0,
            1.0,
        )
        card_x = min(step_x + 14.0, plot.x + plot.width - 188.0)
        card_y = max(plot.y + 28.0, step_y - 26.0 + offset)
        parts.append(
            svg_rect(
                card_x,
                card_y,
                174.0,
                42.0,
                fill=result.soft,
                stroke=result.color,
                stroke_width=1.2,
                rx=12.0,
                opacity=0.95,
            )
        )
        parts.append(
            svg_text(
                card_x + 12.0,
                card_y + 17.0,
                f"{result.display_name}: ODE @ step {result.ode_start_step}",
                size=12,
                weight=700,
                fill=result.accent,
            )
        )
        parts.append(
            svg_text(
                card_x + 12.0,
                card_y + 32.0,
                f"t={fmt(result.ode_start_time)} "
                f"(threshold {fmt(result.ode_time_duration)})",
                size=11,
                fill="#475467",
            )
        )


def render_dt_panel(
    parts: list[str],
    panel: Panel,
    results: list[CaseMetrics],
    num_steps: int,
) -> None:
    draw_panel_frame(parts, panel, "Per-step |Delta t|")
    max_dt = max(result.max_abs_dt for result in results)
    y_top = np.ceil(max_dt * 1000.0) / 1000.0 + 0.001
    ticks = np.linspace(0.0, y_top, 6)
    plot = draw_axes(
        parts,
        panel,
        x_ticks=[0, 50, 100, 150, 199],
        y_ticks=[float(value) for value in ticks],
        x_domain=(0.0, float(num_steps - 1)),
        y_domain=(0.0, float(y_top)),
        x_label="Sampling step index",
        y_label="Absolute step size",
    )

    for result in results:
        points: list[tuple[float, float]] = []
        for step_idx, value in enumerate(result.abs_dt):
            x, y = panel_coords(
                plot,
                float(step_idx),
                0.0,
                float(num_steps - 1),
                float(value),
                0.0,
                float(y_top),
            )
            points.append((x, y))
        parts.append(
            svg_polyline(
                points,
                stroke=result.color,
                stroke_width=3.2,
            )
        )
        marker_x, marker_y = panel_coords(
            plot,
            float(result.ode_start_step),
            0.0,
            float(num_steps - 1),
            float(result.abs_dt[result.ode_start_step]),
            0.0,
            float(y_top),
        )
        parts.append(
            svg_circle(
                marker_x,
                marker_y,
                4.8,
                fill=result.color,
                stroke="#FFFFFF",
                stroke_width=1.4,
            )
        )


def render_phase_panel(
    parts: list[str],
    panel: Panel,
    results: list[CaseMetrics],
    configs: list[ScheduleConfig],
    fields: list[str],
    num_steps: int,
) -> None:
    draw_panel_frame(parts, panel, "Phase allocation by step")
    bar_x = panel.x + 110.0
    bar_width = panel.width - 250.0
    bar_height = 34.0
    row_gap = 58.0
    top_y = panel.y + 74.0

    parts.append(
        svg_text(
            bar_x,
            panel.y + 48.0,
            phase_panel_caption(configs, fields),
            size=12,
            fill="#667085",
        )
    )

    for row_idx, result in enumerate(results):
        y = top_y + row_idx * row_gap
        parts.append(
            svg_text(
                panel.x + 20.0,
                y + 22.0,
                result.display_name,
                size=13,
                weight=700,
                fill=result.accent,
            )
        )
        parts.append(
            svg_rect(
                bar_x,
                y,
                bar_width,
                bar_height,
                fill="#F8FAFC",
                stroke="#D6DCE5",
                stroke_width=1.0,
                rx=17.0,
            )
        )

        churn_width = bar_width * result.churn_steps / num_steps
        middle_width = bar_width * result.middle_steps / num_steps
        ode_width = bar_width * result.ode_steps / num_steps

        parts.append(
            svg_rect(
                bar_x,
                y,
                churn_width,
                bar_height,
                fill=PHASE_COLORS["churn"],
                rx=17.0,
            )
        )
        parts.append(
            svg_rect(
                bar_x + churn_width,
                y,
                middle_width,
                bar_height,
                fill=PHASE_COLORS["middle"],
            )
        )
        parts.append(
            svg_rect(
                bar_x + churn_width + middle_width,
                y,
                ode_width,
                bar_height,
                fill=result.color,
                rx=17.0,
            )
        )
        marker_x = bar_x + bar_width * result.ode_start_step / num_steps
        parts.append(
            svg_line(
                marker_x,
                y - 4.0,
                marker_x,
                y + bar_height + 4.0,
                stroke=result.accent,
                stroke_width=2.0,
                dash="5 5",
            )
        )
        parts.append(
            svg_text(
                bar_x + bar_width + 10.0,
                y + 16.0,
                f"ODE step {result.ode_start_step}",
                size=12,
                fill="#344054",
            )
        )
        parts.append(
            svg_text(
                bar_x + bar_width + 10.0,
                y + 31.0,
                f"{result.ode_steps} ODE steps ({fmt(result.ode_fraction * 100, 0)}%)",
                size=11,
                fill="#667085",
            )
        )

    legend_y = panel.y + panel.height - 26.0
    legend_x = panel.x + 20.0
    for label, key in [
        ("Churn / early SDE", "churn"),
        ("Middle SDE", "middle"),
        ("Late deterministic ODE", "ode"),
    ]:
        parts.append(
            svg_rect(
                legend_x,
                legend_y - 10.0,
                14.0,
                14.0,
                fill=PHASE_COLORS[key],
                stroke="none",
                rx=4.0,
            )
        )
        parts.append(
            svg_text(
                legend_x + 22.0,
                legend_y + 2.0,
                label,
                size=12,
                fill="#475467",
            )
        )
        legend_x += 128.0 if key == "churn" else 96.0


def render_summary_panel(
    parts: list[str],
    panel: Panel,
    results: list[CaseMetrics],
    configs: list[ScheduleConfig],
    fields: list[str],
) -> None:
    draw_panel_frame(parts, panel, "Readout")
    headline, subline, note = summary_copy(results, configs, fields)
    parts.append(
        svg_text(
            panel.x + 22.0,
            panel.y + 56.0,
            headline,
            size=14,
            weight=600,
            fill="#0F172A",
        )
    )

    parts.append(
        svg_text(
            panel.x + 22.0,
            panel.y + 80.0,
            subline,
            size=12,
            fill="#475467",
        )
    )
    if note is not None:
        parts.append(
            svg_text(
                panel.x + 22.0,
                panel.y + 98.0,
                note,
                size=12,
                fill="#667085",
            )
        )

    table_x = panel.x + 22.0
    table_y = panel.y + (120.0 if note is not None else 102.0)
    row_height = 28.0
    columns = [
        ("Case", 120.0),
        ("ODE start", 150.0),
        ("ODE steps", 130.0),
        ("Mean |dt| pre-ODE", 170.0),
        ("Mean |dt| ODE", 150.0),
        ("ODE steps / unit t", 170.0),
    ]

    cursor_x = table_x
    for column_name, column_width in columns:
        parts.append(
            svg_rect(
                cursor_x,
                table_y,
                column_width,
                row_height,
                fill="#EEF2F8",
                stroke="#D6DCE5",
                stroke_width=1.0,
            )
        )
        parts.append(
            svg_text(
                cursor_x + 10.0,
                table_y + 18.0,
                column_name,
                size=12,
                weight=700,
                fill="#344054",
            )
        )
        cursor_x += column_width

    for row_idx, result in enumerate(results, start=1):
        cursor_x = table_x
        row_y = table_y + row_idx * row_height
        fill = "#FFFFFF" if row_idx % 2 else "#FAFBFC"
        row_values = [
            result.display_name,
            f"{result.ode_start_step} ({fmt(result.ode_start_step_fraction * 100, 1)}%)",
            f"{result.ode_steps} ({fmt(result.ode_fraction * 100, 0)}%)",
            fmt(result.mean_abs_dt_pre_ode, 4),
            fmt(result.mean_abs_dt_ode, 4),
            fmt(result.steps_per_unit_ode, 1),
        ]
        for value, (_, column_width) in zip(row_values, columns, strict=True):
            parts.append(
                svg_rect(
                    cursor_x,
                    row_y,
                    column_width,
                    row_height,
                    fill=fill,
                    stroke="#E1E7F0",
                    stroke_width=1.0,
                )
            )
            text_fill = result.accent if cursor_x == table_x else "#344054"
            text_weight = 700 if cursor_x == table_x else 400
            parts.append(
                svg_text(
                    cursor_x + 10.0,
                    row_y + 18.0,
                    value,
                    size=12,
                    weight=text_weight,
                    fill=text_fill,
                )
            )
            cursor_x += column_width


def render_svg(
    results: list[CaseMetrics],
    configs: list[ScheduleConfig],
    title: str,
) -> str:
    fields = varied_fields(configs)
    width = 1600.0
    height = 1240.0
    parts: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" '
            f'height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">'
        ),
        "<defs>",
        (
            '<linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">'
            '<stop offset="0%" stop-color="#F7F4EE" />'
            '<stop offset="100%" stop-color="#EEF4FA" />'
            "</linearGradient>"
        ),
        (
            '<filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">'
            '<feDropShadow dx="0" dy="8" stdDeviation="14" flood-color="#0F172A" '
            'flood-opacity="0.08" />'
            "</filter>"
        ),
        "</defs>",
        svg_rect(0.0, 0.0, width, height, fill="url(#bg)"),
    ]

    parts.append(
        svg_text(
            70.0,
            66.0,
            title,
            size=28,
            weight=800,
            fill="#0F172A",
        )
    )
    if set(fields).issubset({"ode_time_duration", "sampling_schedule_ode_fraction"}):
        subtitle = (
            "Three-way comparison with all other knobs fixed. "
            "Focus: when the late deterministic ODE phase actually starts."
        )
    elif set(fields).issubset(
        {"churn_until_time", "sampling_schedule_churn_fraction", "churn_factor"}
    ):
        subtitle = (
            "Churn-window candidate comparison from the Obsidian note. "
            "Focus: how a wider early churn region reallocates the step budget."
        )
    else:
        subtitle = "Three-way phase-power schedule comparison with fixed base config."
    parts.append(
        svg_text(
            70.0,
            94.0,
            subtitle,
            size=15,
            fill="#475467",
        )
    )

    legend_x = 70.0
    for result, cfg in zip(results, configs, strict=True):
        parts.append(
            svg_rect(
                legend_x,
                108.0,
                16.0,
                16.0,
                fill=result.color,
                rx=5.0,
            )
        )
        parts.append(
            svg_text(
                legend_x + 24.0,
                121.0,
                f"{result.display_name}: {case_setting_summary(cfg, fields)}",
                size=12,
                fill="#344054",
            )
        )
        legend_x += 410.0

    schedule_panel = Panel(70.0, 144.0, 1460.0, 430.0)
    dt_panel = Panel(70.0, 604.0, 860.0, 310.0)
    phase_panel = Panel(960.0, 604.0, 570.0, 310.0)
    summary_panel = Panel(70.0, 944.0, 1460.0, 240.0)

    for panel in [schedule_panel, dt_panel, phase_panel, summary_panel]:
        shadow_rect = svg_rect(
            panel.x,
            panel.y,
            panel.width,
            panel.height,
            fill="#FFFFFF",
            stroke="none",
            rx=22.0,
        )
        parts.append(f'<g filter="url(#shadow)">{shadow_rect}</g>')

    render_schedule_panel(parts, schedule_panel, results, configs[0].num_steps)
    render_dt_panel(parts, dt_panel, results, configs[0].num_steps)
    render_phase_panel(parts, phase_panel, results, configs, fields, configs[0].num_steps)
    render_summary_panel(parts, summary_panel, results, configs, fields)

    parts.append("</svg>")
    return "\n".join(parts)


def save_summary_csv(path: Path, results: list[CaseMetrics]) -> None:
    rows = [
        {
            "case": result.name,
            "display_name": result.display_name,
            "ode_time_duration": result.ode_time_duration,
            "ode_fraction": result.ode_fraction,
            "churn_until_time": result.churn_until_time,
            "churn_fraction": result.churn_fraction,
            "churn_factor": result.churn_factor,
            "ode_start_step": result.ode_start_step,
            "ode_start_step_fraction": result.ode_start_step_fraction,
            "ode_start_time": result.ode_start_time,
            "ode_prev_time": result.ode_prev_time,
            "churn_steps": result.churn_steps,
            "middle_steps": result.middle_steps,
            "ode_steps": result.ode_steps,
            "mean_abs_dt_total": result.mean_abs_dt_total,
            "mean_abs_dt_pre_ode": result.mean_abs_dt_pre_ode,
            "mean_abs_dt_ode": result.mean_abs_dt_ode,
            "max_abs_dt": result.max_abs_dt,
            "min_abs_dt": result.min_abs_dt,
            "steps_per_unit_pre_ode": result.steps_per_unit_pre_ode,
            "steps_per_unit_ode": result.steps_per_unit_ode,
        }
        for result in results
    ]
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary_json(
    path: Path,
    cfg: ScheduleConfig,
    results: list[CaseMetrics],
) -> None:
    payload = {
        "base_config": asdict(cfg),
        "cases": [asdict(result) for result in results],
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def local_tag(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", maxsplit=1)[-1]
    return tag


def hex_to_rgba(color: str, opacity: float = 1.0) -> tuple[int, int, int, int]:
    rgb = ImageColor.getrgb(color)
    alpha = max(0, min(255, round(255 * opacity)))
    return rgb[0], rgb[1], rgb[2], alpha


def draw_diagonal_gradient(
    width: int,
    height: int,
    color_start: str,
    color_end: str,
) -> Image.Image:
    start = np.array(ImageColor.getrgb(color_start), dtype=np.float32)
    end = np.array(ImageColor.getrgb(color_end), dtype=np.float32)
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    blend = ((x + y) / 2.0)[..., None]
    rgb = start + (end - start) * blend
    alpha = np.full((height, width, 1), 255, dtype=np.float32)
    rgba = np.concatenate([rgb, alpha], axis=-1).clip(0, 255).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def font_path(weight: int) -> str | None:
    candidates = []
    if weight >= 600:
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
        if Path(candidate).exists():
            return candidate
    return None


def load_font(size: int, weight: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = font_path(weight)
    if path is None:
        return ImageFont.load_default()
    return ImageFont.truetype(path, size=size)


def draw_overlay_rect(
    image: Image.Image,
    box: tuple[float, float, float, float],
    *,
    fill: str | None,
    stroke: str | None,
    stroke_width: float,
    radius: float,
    opacity: float,
) -> None:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    coords = tuple(int(round(value)) for value in box)
    line_width = max(1, int(round(stroke_width))) if stroke else 0
    kwargs: dict[str, Any] = {
        "xy": coords,
        "radius": max(0, int(round(radius))),
    }
    if fill is not None:
        kwargs["fill"] = hex_to_rgba(fill, opacity)
    if stroke is not None and line_width > 0:
        kwargs["outline"] = hex_to_rgba(stroke, opacity)
        kwargs["width"] = line_width
    draw.rounded_rectangle(**kwargs)
    image.alpha_composite(overlay)


def draw_overlay_line(
    image: Image.Image,
    xy: tuple[float, float, float, float],
    *,
    stroke: str,
    stroke_width: float,
    opacity: float,
    dash: str | None = None,
) -> None:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    x1, y1, x2, y2 = xy
    width = max(1, int(round(stroke_width)))
    fill = hex_to_rgba(stroke, opacity)
    if dash is None:
        draw.line((x1, y1, x2, y2), fill=fill, width=width)
    else:
        dash_values = [float(value) for value in dash.split()]
        if len(dash_values) == 1:
            dash_values = [dash_values[0], dash_values[0]]
        pattern = dash_values[:2]
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length == 0.0:
            return
        dx = (x2 - x1) / length
        dy = (y2 - y1) / length
        position = 0.0
        draw_on = True
        pattern_index = 0
        while position < length:
            segment = pattern[pattern_index % len(pattern)]
            next_position = min(length, position + segment)
            if draw_on:
                sx = x1 + dx * position
                sy = y1 + dy * position
                ex = x1 + dx * next_position
                ey = y1 + dy * next_position
                draw.line((sx, sy, ex, ey), fill=fill, width=width)
            position = next_position
            pattern_index += 1
            draw_on = not draw_on
    image.alpha_composite(overlay)


def draw_overlay_circle(
    image: Image.Image,
    center: tuple[float, float],
    radius: float,
    *,
    fill: str,
    stroke: str,
    stroke_width: float,
) -> None:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    cx, cy = center
    bounds = (
        int(round(cx - radius)),
        int(round(cy - radius)),
        int(round(cx + radius)),
        int(round(cy + radius)),
    )
    draw.ellipse(
        bounds,
        fill=hex_to_rgba(fill, 1.0),
        outline=hex_to_rgba(stroke, 1.0),
        width=max(1, int(round(stroke_width))),
    )
    image.alpha_composite(overlay)


def draw_overlay_polyline(
    image: Image.Image,
    points: list[tuple[float, float]],
    *,
    stroke: str,
    stroke_width: float,
    opacity: float,
) -> None:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.line(
        points,
        fill=hex_to_rgba(stroke, opacity),
        width=max(1, int(round(stroke_width))),
        joint="curve",
    )
    image.alpha_composite(overlay)


def draw_svg_text(
    image: Image.Image,
    *,
    x: float,
    y: float,
    text: str,
    size: int,
    weight: int,
    fill: str,
    anchor: str,
) -> None:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = load_font(size, weight)
    if isinstance(font, ImageFont.FreeTypeFont):
        ascent, _ = font.getmetrics()
    else:
        ascent = size
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    x_left = x
    if anchor == "middle":
        x_left = x - text_width / 2.0
    elif anchor == "end":
        x_left = x - text_width
    y_top = y - ascent
    draw.text((x_left, y_top), text, font=font, fill=hex_to_rgba(fill, 1.0))
    image.alpha_composite(overlay)


def rasterize_svg_to_png(svg_text: str, png_path: Path, scale: int) -> None:
    root = ET.fromstring(svg_text)
    width = int(float(root.attrib["width"]))
    height = int(float(root.attrib["height"]))
    raster_scale = max(1, scale)
    scaled_width = width * raster_scale
    scaled_height = height * raster_scale

    image = draw_diagonal_gradient(
        scaled_width,
        scaled_height,
        "#F7F4EE",
        "#EEF4FA",
    )

    def visit(element: ET.Element) -> None:
        tag = local_tag(element.tag)
        if tag in {"svg", "defs"}:
            for child in element:
                visit(child)
            return
        if tag == "g":
            for child in element:
                visit(child)
            return
        if tag == "rect":
            fill = element.attrib.get("fill")
            if fill == "url(#bg)":
                return
            stroke = element.attrib.get("stroke")
            draw_overlay_rect(
                image,
                (
                    float(element.attrib.get("x", 0.0)) * raster_scale,
                    float(element.attrib.get("y", 0.0)) * raster_scale,
                    (
                        float(element.attrib.get("x", 0.0))
                        + float(element.attrib.get("width", 0.0))
                    )
                    * raster_scale,
                    (
                        float(element.attrib.get("y", 0.0))
                        + float(element.attrib.get("height", 0.0))
                    )
                    * raster_scale,
                ),
                fill=None if fill in {None, "none"} else fill,
                stroke=None if stroke in {None, "none"} else stroke,
                stroke_width=float(element.attrib.get("stroke-width", 0.0))
                * raster_scale,
                radius=float(element.attrib.get("rx", 0.0)) * raster_scale,
                opacity=float(element.attrib.get("opacity", 1.0)),
            )
            return
        if tag == "line":
            draw_overlay_line(
                image,
                (
                    float(element.attrib["x1"]) * raster_scale,
                    float(element.attrib["y1"]) * raster_scale,
                    float(element.attrib["x2"]) * raster_scale,
                    float(element.attrib["y2"]) * raster_scale,
                ),
                stroke=element.attrib["stroke"],
                stroke_width=float(element.attrib.get("stroke-width", 1.0))
                * raster_scale,
                opacity=float(element.attrib.get("opacity", 1.0)),
                dash=element.attrib.get("stroke-dasharray"),
            )
            return
        if tag == "circle":
            draw_overlay_circle(
                image,
                (
                    float(element.attrib["cx"]) * raster_scale,
                    float(element.attrib["cy"]) * raster_scale,
                ),
                float(element.attrib["r"]) * raster_scale,
                fill=element.attrib["fill"],
                stroke=element.attrib["stroke"],
                stroke_width=float(element.attrib.get("stroke-width", 1.0))
                * raster_scale,
            )
            return
        if tag == "polyline":
            points: list[tuple[float, float]] = []
            for point_text in element.attrib["points"].split():
                x_value, y_value = point_text.split(",")
                points.append(
                    (
                        float(x_value) * raster_scale,
                        float(y_value) * raster_scale,
                    )
                )
            draw_overlay_polyline(
                image,
                points,
                stroke=element.attrib["stroke"],
                stroke_width=float(element.attrib.get("stroke-width", 1.0))
                * raster_scale,
                opacity=float(element.attrib.get("opacity", 1.0)),
            )
            return
        if tag == "text":
            draw_svg_text(
                image,
                x=float(element.attrib["x"]) * raster_scale,
                y=float(element.attrib["y"]) * raster_scale,
                text="".join(element.itertext()),
                size=max(
                    1, int(round(float(element.attrib["font-size"]) * raster_scale))
                ),
                weight=int(element.attrib.get("font-weight", 400)),
                fill=element.attrib.get("fill", "#000000"),
                anchor=element.attrib.get("text-anchor", "start"),
            )
            return

    visit(root)

    if raster_scale > 1:
        image = image.resize((width, height), resample=Image.Resampling.LANCZOS)
    image.save(png_path)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = load_base_config(args.config.resolve(), args.num_steps)
    base_cfg.validate()

    case_specs = args.case or DEFAULT_CASES
    results: list[CaseMetrics] = []
    case_configs: list[ScheduleConfig] = []
    for spec in case_specs:
        name, overrides = parse_case_spec(spec)
        case_cfg = apply_overrides(base_cfg, overrides)
        times = build_phase_power_schedule(case_cfg)
        case_configs.append(case_cfg)
        results.append(summarize_case(name, case_cfg, times))

    svg_path = out_dir / "ecsi_time_step_comparison.svg"
    png_path = out_dir / "ecsi_time_step_comparison.png"
    csv_path = out_dir / "ecsi_time_step_summary.csv"
    json_path = out_dir / "ecsi_time_step_summary.json"

    svg_text = render_svg(results, case_configs, args.title)
    svg_path.write_text(svg_text, encoding="utf-8")
    rasterize_svg_to_png(svg_text, png_path, args.png_scale)
    save_summary_csv(csv_path, results)
    save_summary_json(json_path, base_cfg, results)

    print(f"SVG saved to: {svg_path}")
    print(f"PNG saved to: {png_path}")
    print(f"CSV saved to: {csv_path}")
    print(f"JSON saved to: {json_path}")
    for result in results:
        print(
            f"{result.display_name}: ODE starts at step {result.ode_start_step} "
            f"({result.ode_start_step_fraction * 100:.1f}%), "
            f"{result.ode_steps} ODE steps"
        )


if __name__ == "__main__":
    main()
