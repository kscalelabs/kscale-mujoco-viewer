"""Plotting utilities for the MuJoCo viewer."""

import itertools
import logging
import platform
from typing import Callable, ParamSpec, TypeVar

import dearpygui.dearpygui as dpg
import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar("T")
P = ParamSpec("P")


def run_on_main_thread(func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> None:
    """Execute function on the main thread if on macOS, otherwise execute directly.

    This is needed for DearPyGui operations on macOS, which must be performed on the main thread
    to avoid crashes with SIGTRAP/trace trap related to thread assertions in Apple's frameworks.
    """
    if platform.system() == "Darwin":  # macOS
        try:
            from Foundation import NSThread

            if not NSThread.isMainThread():
                from PyObjCTools import AppHelper

                # Use callAfter to schedule the function on the main thread
                AppHelper.callAfter(func, *args, **kwargs)
                return
        except ImportError:
            # Fall back to direct execution if PyObjC is not available
            pass
    # Direct execution for non-macOS or if PyObjC is not available
    func(*args, **kwargs)


class Plot:
    def __init__(
        self,
        name: str,
        group_name: str,
        x_label: str,
        y_label: str,
        y_axis_min: float | None = None,
        y_axis_max: float | None = None,
    ) -> None:
        self.name = name
        self.group_name = group_name
        self.series_tag = f"series_{name}"
        self.x_label = x_label
        self.y_label = y_label
        self.y_axis_min = y_axis_min
        self.y_axis_max = y_axis_max
        self.x_data: list[float] = []
        self.y_data: list[float] = []


class Plotter:
    def __init__(self, window_title: str = "Plotter", window_width: int = 1000, window_height: int = 600) -> None:
        self.plots: dict[str, Plot] = {}  # Dictionary to track plot data
        self.plot_count: int = 0  # Counter for dynamic plots
        self.visible_points: int = 200  # Number of recent points to show when auto-fitting
        self.plot_groups: dict[str, list[str]] = {}  # Dictionary to track plot groups
        self.group_index_mappings: dict[str, dict[int, str]] = {}

        dpg.create_context()
        # Create a main window to hold all plots
        with dpg.window(label=window_title, tag="main_window"):
            # Create a horizontal group for columns (plot groups)
            dpg.add_group(tag="plot_columns", horizontal=True)

        dpg.create_viewport(title=window_title, width=window_width, height=window_height)
        dpg.setup_dearpygui()
        dpg.set_primary_window("main_window", True)

    def add_plot_group(
        self,
        group_name: str,
        index_mapping: dict[int, str] | None = None,
        y_axis_min: float | None = None,
        y_axis_max: float | None = None,
    ) -> str:
        """Add a new group (column) for plots."""
        if group_name in self.plot_groups:
            return group_name  # Group already exists

        # Create a vertical group (column) for this group's plots
        with dpg.group(parent="plot_columns", tag=f"group_column_{group_name}"):
            # Add a title for the group
            dpg.add_text(f"== {group_name.upper()} ==")

        self.plot_groups[group_name] = []
        if index_mapping is not None:
            self.group_index_mappings[group_name] = index_mapping
            for idx, plot_name in index_mapping.items():
                self.add_plot(plot_name, group=group_name, y_axis_min=y_axis_min, y_axis_max=y_axis_max)
        return group_name

    def add_plot(
        self,
        plot_name: str,
        x_label: str = "Total Sim Time",
        y_label: str = "y",
        y_axis_min: float | None = None,
        y_axis_max: float | None = None,
        group: str | None = None,
    ) -> None:
        """Add a new plot, optionally to a specific group."""
        # If no group specified, use default
        if group is None:
            group = "default"

        # Create the group if it doesn't exist
        if group not in self.plot_groups:
            self.add_plot_group(group)

        # Add the plot to the specified group
        parent_tag = f"group_column_{group}"

        # Add a new plot inside the group column
        with dpg.group(parent=parent_tag, tag=f"plot_container_{plot_name}"):
            with dpg.plot(label=plot_name, width=500, height=200, tag=f"plot_{plot_name}"):
                dpg.add_plot_legend()
                # Create the axes with unique tags.
                dpg.add_plot_axis(dpg.mvXAxis, label=x_label, tag=f"x_axis_{plot_name}")
                dpg.add_plot_axis(dpg.mvYAxis, label=y_label, tag=f"y_axis_{plot_name}")
                if y_axis_min is not None and y_axis_max is not None:
                    dpg.set_axis_limits(f"y_axis_{plot_name}", y_axis_min, y_axis_max)
                else:
                    dpg.set_axis_limits_auto(f"y_axis_{plot_name}")
                # Create an empty line series to be updated later.
                dpg.add_line_series([], [], label=plot_name, parent=f"y_axis_{plot_name}", tag=f"series_{plot_name}")

            with dpg.group(horizontal=True):
                default_fit_y_axis = y_axis_min is None and y_axis_max is None
                dpg.add_checkbox(
                    label="Auto-fit x-axis", tag=f"auto_fit_checkbox_x_axis_{plot_name}", default_value=True
                )
                dpg.add_checkbox(
                    label="Manual limits y-axis",
                    tag=f"manual_limits_checkbox_y_axis_{plot_name}",
                    default_value=not default_fit_y_axis,
                )
                dpg.add_checkbox(
                    label="Auto-fit y-axis",
                    tag=f"auto_fit_checkbox_y_axis_{plot_name}",
                    default_value=default_fit_y_axis,
                )

        # Initialize empty data lists for the new plot.
        self.plots[plot_name] = Plot(plot_name, group, x_label, y_label, y_axis_min, y_axis_max)
        self.plot_groups[group].append(plot_name)
        logger.info("Added plot: %s to group: %s", plot_name, group)

    def _update_plot_axes(self, plot_name: str) -> None:
        """Update both x and y axis limits based on auto-fit settings.

        Args:
            plot_name: Name of the plot
        """
        # Handle X-axis
        if dpg.get_value(f"auto_fit_checkbox_x_axis_{plot_name}"):
            x_data = self.plots[plot_name].x_data
            if len(x_data) > self.visible_points:
                # For x-axis with many points, show only most recent ones
                x_min = x_data[-self.visible_points]
                x_max = x_data[-1]
                # Add a small margin (5% of range)
                margin = (x_max - x_min) * 0.05
                dpg.set_axis_limits(f"x_axis_{plot_name}", x_min - margin, x_max + margin)
            else:
                # Otherwise fit all data
                dpg.fit_axis_data(f"x_axis_{plot_name}")
        else:
            # If auto-fit is disabled, make sure the axis isn't locked
            dpg.set_axis_limits_auto(f"x_axis_{plot_name}")

        # Handle Y-axis
        if (
            dpg.get_value(f"manual_limits_checkbox_y_axis_{plot_name}")
            and self.plots[plot_name].y_axis_min is not None
            and self.plots[plot_name].y_axis_max is not None
        ):
            dpg.set_axis_limits(
                f"y_axis_{plot_name}", self.plots[plot_name].y_axis_min, self.plots[plot_name].y_axis_max
            )
        elif dpg.get_value(f"auto_fit_checkbox_y_axis_{plot_name}"):
            dpg.fit_axis_data(f"y_axis_{plot_name}")
        else:
            dpg.set_axis_limits_auto(f"y_axis_{plot_name}")

    def update_axes(self) -> None:
        for plot_name in self.plots:
            self._update_plot_axes(plot_name)

    def update_plot(self, plot_name: str, x: float, y: float) -> None:
        if plot_name in self.plots:
            self.plots[plot_name].x_data.append(x)
            self.plots[plot_name].y_data.append(y)
            # Update the corresponding line series.
            dpg.set_value(
                self.plots[plot_name].series_tag, [self.plots[plot_name].x_data, self.plots[plot_name].y_data]
            )

            # Update both axis limits
            self._update_plot_axes(plot_name)
        else:
            logger.error("Plot '%s' not found!", plot_name)

    def update_plot_group(self, group_name: str, x_value: float, y_values: list[float]) -> None:
        if group_name in self.group_index_mappings:
            index_mapping = self.group_index_mappings[group_name]
            for i, y_value in enumerate(y_values):
                # breakpoint()
                plot_name = index_mapping[i]
                self.update_plot(plot_name, x_value, y_value)

    def _on_viewport_ready(self) -> None:
        logger.info("Viewport is ready.")
        run_on_main_thread(dpg.render_dearpygui_frame)

    def render_frame(self) -> None:
        if not dpg.is_viewport_ok():
            logger.warning("Viewport is not created yet. Waiting for it to be ready.")
            dpg.set_viewport_resize_callback(self._on_viewport_ready)
        else:
            run_on_main_thread(dpg.render_dearpygui_frame)

    def start(self) -> None:
        run_on_main_thread(dpg.show_viewport)

    def close(self) -> None:
        run_on_main_thread(dpg.destroy_context)


if __name__ == "__main__":
    # Example usage:
    plotter = Plotter("Robot Simulation", 1200, 800)

    # Add plots in different groups
    plotter.add_plot("joint1_position", y_label="Position", group="Joints")
    plotter.add_plot("joint1_velocity", y_label="Velocity", group="Joints")

    plotter.add_plot("reward_total", y_label="Total Reward", group="Rewards")
    plotter.add_plot("reward_height", y_label="Height Reward", group="Rewards")

    plotter.add_plot("torso_height", y_label="Height", group="Body")
    plotter.add_plot("head_height", y_label="Height", group="Body")
    plotter.start()

    # Simulate data updates
    for i in itertools.count():
        time_val = i * 0.1
        plotter.update_plot("joint1_position", time_val, np.sin(time_val))
        plotter.update_plot("joint1_velocity", time_val, np.cos(time_val))
        plotter.update_plot("reward_total", time_val, 0.8 + 0.2 * np.sin(time_val))
        plotter.update_plot("reward_height", time_val, 0.5 + 0.5 * np.sin(time_val))
        plotter.update_plot("torso_height", time_val, 1.0 + 0.1 * np.sin(time_val))
        plotter.update_plot("head_height", time_val, 1.5 + 0.1 * np.sin(time_val))
        plotter.render_frame()

    plotter.close()
