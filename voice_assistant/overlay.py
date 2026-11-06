"""On-screen status pill: listening / thinking / speaking.

Runs as a separate process under the *system* Python (GTK and gtk-layer-shell come
from the distro, not from the venv) and reads JSON lines on stdin:

    {"state": "attentive", "seconds": 8, "name": "Джарвиз", "tasks": []}

It exits when stdin closes. Drag the panel to move it; right click resets its
position. Standalone: GTK is provided by the system Python, not the project venv.
"""

from __future__ import annotations

import json
import math
import sys
import threading
import time
import importlib.util
import subprocess
from pathlib import Path

import cairo
import gi

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
gi.require_version("GtkLayerShell", "0.1")
gi.require_version("PangoCairo", "1.0")
from gi.repository import Gdk, GLib, Gtk, GtkLayerShell, Pango, PangoCairo  # noqa: E402

spec = importlib.util.spec_from_file_location("overlay_model", Path(__file__).with_name("overlay_model.py"))
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)

# state: (dot colour, label)
STATES = {
    "idle": ((0.62, 0.64, 0.70), ""),
    "hearing": ((0.92, 0.93, 0.96), ""),
    "attentive": ((0.31, 0.55, 1.0), "Слушаю"),
    "thinking": ((0.96, 0.73, 0.26), "Думаю"),
    "speaking": ((0.24, 0.81, 0.56), "Говорю"),
    "muted": ((1.0, 0.36, 0.36), "Микрофон выключен"),
}
HEIGHT = 34
DOT = 6
PAD = 13
FONT = Pango.FontDescription.from_string("Sans 10.5")


class Pill(Gtk.Window):
    def __init__(self, position: str, settings_path: str = "", name: str = "") -> None:
        super().__init__()
        self.state = "idle"
        self.badge = ""
        self.until = 0.0  # attentive countdown end (monotonic)
        self.total = 0.0
        self.changed = time.monotonic()
        self.width = HEIGHT
        self.height = HEIGHT
        self.tasks = []
        self.name = name
        self.settings_path = settings_path
        self.position = position
        self.saved = model.load_position(settings_path)
        self.x, self.y = self.saved or (0, 0)
        self.drag = None

        GtkLayerShell.init_for_window(self)
        GtkLayerShell.set_layer(self, GtkLayerShell.Layer.OVERLAY)
        GtkLayerShell.set_namespace(self, "nora-indicator")
        GtkLayerShell.set_anchor(self, GtkLayerShell.Edge.TOP, True)
        GtkLayerShell.set_anchor(self, GtkLayerShell.Edge.LEFT, True)
        display = Gdk.Display.get_default()
        self.monitor = display.get_primary_monitor() or display.get_monitor(0)
        GtkLayerShell.set_monitor(self, self.monitor)
        GtkLayerShell.set_keyboard_mode(self, GtkLayerShell.KeyboardMode.NONE)
        GtkLayerShell.set_exclusive_zone(self, -1)

        visual = self.get_screen().get_rgba_visual()
        if visual:
            self.set_visual(visual)
        self.set_app_paintable(True)
        self.area = Gtk.DrawingArea()
        self.area.connect("draw", self.draw)
        self.area.add_events(Gdk.EventMask.BUTTON_PRESS_MASK | Gdk.EventMask.BUTTON_RELEASE_MASK | Gdk.EventMask.POINTER_MOTION_MASK)
        self.area.connect("button-press-event", self.press)
        self.area.connect("button-release-event", self.release)
        self.area.connect("motion-notify-event", self.motion)
        self.area.set_tooltip_text("Перетащи, чтобы переместить. Правый клик — вернуть на исходное место.")
        self.add(self.area)
        self.resize_to_content()
        GLib.timeout_add(33, self.tick)

    def update(self, message: dict) -> bool:
        state = message.get("state", self.state)
        if state != self.state:
            self.changed = time.monotonic()
        self.state = state if state in STATES else "idle"
        self.badge = message.get("badge", "")
        self.name = str(message.get("name", self.name))
        self.tasks = [t for t in message.get("tasks", []) if isinstance(t, dict)][:3]
        hint = "Перетащи, чтобы переместить. Правый клик — вернуть на исходное место."
        details = [f"{t.get('title', 'Задача')}\n{t.get('detail', '')}".strip() for t in self.tasks]
        self.area.set_tooltip_text("\n\n".join([hint, *details]))
        seconds = float(message.get("seconds") or 0)
        if seconds:
            self.until, self.total = time.monotonic() + seconds, seconds
        self.resize_to_content()
        return False

    def label(self) -> str:
        text = STATES[self.state][1]
        return " · ".join(t for t in (self.name, text, "" if self.tasks else self.badge) if t)

    def place(self) -> None:
        geometry = self.monitor.get_geometry()
        if self.saved is None:
            self.x = (geometry.width - self.width) // 2
            self.y = 56 if self.position == "top" else geometry.height - self.height - 26
        self.x, self.y = model.clamp_position(self.x, self.y, self.width, self.height, geometry.width, geometry.height)
        GtkLayerShell.set_margin(self, GtkLayerShell.Edge.LEFT, self.x)
        GtkLayerShell.set_margin(self, GtkLayerShell.Edge.TOP, self.y)

    def cursor(self, event) -> tuple[float, float]:
        # Wayland does not expose global pointer coordinates; Hyprland does.
        try:
            data = json.loads(subprocess.check_output(["hyprctl", "cursorpos", "-j"], timeout=0.15))
            return float(data["x"]), float(data["y"])
        except (OSError, ValueError, KeyError, subprocess.SubprocessError):
            return self.x + event.x, self.y + event.y

    def press(self, _widget, event) -> bool:
        if event.button == 3:
            self.saved = None
            self.place()
            try:
                Path(self.settings_path).unlink(missing_ok=True) if self.settings_path else None
            except OSError:
                pass
            return True
        if event.button == 1:
            self.drag = (self.cursor(event), self.x, self.y)
            return True
        return False

    def motion(self, _widget, event) -> bool:
        if self.drag is None:
            return False
        (cx, cy), x, y = self.drag
        nx, ny = self.cursor(event)
        self.x, self.y = round(x + nx - cx), round(y + ny - cy)
        self.saved = (self.x, self.y)
        self.place()
        return True

    def release(self, _widget, event) -> bool:
        if event.button == 1 and self.drag is not None:
            self.motion(_widget, event)
            self.drag = None
            try:
                model.save_position(self.settings_path, self.x, self.y)
            except OSError as error:
                print(f"Cannot save indicator position: {error}", file=sys.stderr)
            return True
        return False

    def resize_to_content(self) -> None:
        text = self.label()
        width = HEIGHT
        if text:
            layout = self.create_pango_layout(text)
            layout.set_font_description(FONT)
            width = PAD + DOT * 2 + 10 + layout.get_pixel_size()[0] + PAD + 2
        width = 356 if self.tasks else min(440, max(width + 20, 120 if self.name else HEIGHT))
        height = HEIGHT + (44 * len(self.tasks) + 6 if self.tasks else 0)
        if width != self.width or height != self.height or not self.get_visible():
            self.width, self.height = width, height
            self.area.set_size_request(width, height)
            self.resize(width, height)
            self.place()

    def tick(self) -> bool:
        self.area.queue_draw()
        return True

    def draw(self, _widget, cr: cairo.Context) -> None:
        now = time.monotonic()
        t = now - self.changed
        w, h = self.width, self.height
        cr.set_operator(cairo.OPERATOR_SOURCE)
        cr.set_source_rgba(0, 0, 0, 0)
        cr.paint()
        cr.set_operator(cairo.OPERATOR_OVER)

        quiet = self.state in ("idle", "hearing") and not self.badge and not self.name and not self.tasks
        r = HEIGHT / 2
        cr.new_sub_path()
        cr.arc(w - r, r, r, -math.pi / 2, 0)
        cr.arc(w - r, h - r, r, 0, math.pi / 2)
        cr.arc(r, h - r, r, math.pi / 2, math.pi)
        cr.arc(r, r, r, math.pi, 3 * math.pi / 2)
        cr.close_path()
        cr.set_source_rgba(0.08, 0.09, 0.11, 0.96 if self.tasks else 0.45 if quiet else 0.90)
        cr.fill_preserve()
        cr.set_source_rgba(1, 1, 1, 0.08)
        cr.set_line_width(1)
        cr.stroke()

        (cr_, cg, cb), _ = STATES[self.state]
        cx = r if quiet else PAD + DOT
        cy = r
        if self.state == "thinking":  # spinning arc
            cr.set_source_rgba(cr_, cg, cb, 0.25)
            cr.arc(cx, cy, DOT + 1, 0, 2 * math.pi)
            cr.set_line_width(2.2)
            cr.stroke()
            cr.set_source_rgb(cr_, cg, cb)
            start = t * 5
            cr.arc(cx, cy, DOT + 1, start, start + 1.9)
            cr.stroke()
        elif self.state == "speaking":  # three bouncing bars
            for i in range(3):
                bh = 4 + 8 * abs(math.sin(t * 7 + i * 1.1))
                cr.set_source_rgb(cr_, cg, cb)
                cr.rectangle(cx - 6 + i * 4.5, cy - bh / 2, 3, bh)
                cr.fill()
        else:
            radius, alpha = DOT, 1.0
            if self.state == "idle":
                radius, alpha = DOT - 2, 0.55
            elif self.state == "hearing":
                radius = DOT - 1 + math.sin(t * 9)
            elif self.state == "attentive":
                halo = (t * 1.2) % 1
                cr.set_source_rgba(cr_, cg, cb, 0.45 * (1 - halo))
                cr.arc(cx, cy, DOT + 6 * halo, 0, 2 * math.pi)
                cr.fill()
            elif self.state == "muted":
                alpha = 0.6 + 0.4 * abs(math.sin(t * 2))
            cr.set_source_rgba(cr_, cg, cb, alpha)
            cr.arc(cx, cy, radius, 0, 2 * math.pi)
            cr.fill()

        text = self.label()
        if text and not quiet:
            layout = PangoCairo.create_layout(cr)
            layout.set_font_description(FONT)
            layout.set_text(text, -1)
            _, th = layout.get_pixel_size()
            cr.set_source_rgba(0.95, 0.96, 0.98, 0.95)
            layout.set_width(max(1, w - 65) * Pango.SCALE)
            layout.set_ellipsize(Pango.EllipsizeMode.END)
            cr.move_to(PAD + DOT * 2 + 10, (HEIGHT - th) / 2)
            PangoCairo.show_layout(cr, layout)

        # Small grip makes the movable surface discoverable.
        for dx in (0, 4):
            for dy in (-4, 0, 4):
                cr.set_source_rgba(0.8, 0.82, 0.9, 0.45)
                cr.arc(w - 16 + dx, HEIGHT / 2 + dy, 0.8, 0, 2 * math.pi)
                cr.fill()
        for i, task in enumerate(self.tasks):
            y = HEIGHT + i * 44
            state = task.get("state", "")
            color = (1.0, 0.42, 0.42) if state == "failed" else (0.3, 0.82, 0.58) if state == "done" else (0.5, 0.65, 1.0)
            cr.set_source_rgba(1, 1, 1, 0.08)
            cr.rectangle(13, y, w - 26, 1)
            cr.fill()
            cr.set_source_rgb(*color)
            cr.arc(19, y + 16, 3, 0, 2 * math.pi)
            cr.fill()
            layout = PangoCairo.create_layout(cr)
            layout.set_font_description(Pango.FontDescription.from_string("Sans 9.5"))
            layout.set_width((w - 50) * Pango.SCALE)
            layout.set_ellipsize(Pango.EllipsizeMode.END)
            layout.set_text(str(task.get("title", "Задача")), -1)
            cr.set_source_rgba(0.96, 0.97, 0.99, 0.96)
            cr.move_to(30, y + 6)
            PangoCairo.show_layout(cr, layout)
            end = now if state in ("awaiting_confirmation", "preparing", "running", "checking") else float(task.get("updated", now))
            elapsed = max(0, int(end - float(task.get("created", end))))
            status = f"{task.get('agent', '')} · {model.TASK_LABELS.get(state, state)} · {elapsed // 60:02}:{elapsed % 60:02}"
            layout.set_font_description(Pango.FontDescription.from_string("Sans 8"))
            layout.set_text(status, -1)
            cr.set_source_rgba(0.7, 0.74, 0.82, 0.95)
            cr.move_to(30, y + 24)
            PangoCairo.show_layout(cr, layout)

        if self.state == "attentive" and self.until > now and self.total:  # follow-up countdown
            left = (self.until - now) / self.total
            cr.set_source_rgba(cr_, cg, cb, 0.8)
            cr.rectangle(r, HEIGHT - 3, (w - 2 * r) * left, 2)
            cr.fill()


def read_stdin(pill: Pill) -> None:
    for line in sys.stdin:
        try:
            message = json.loads(line)
        except ValueError:
            continue
        GLib.idle_add(pill.update, message)
    GLib.idle_add(Gtk.main_quit)


def main() -> None:
    position = sys.argv[1] if len(sys.argv) > 1 else "bottom"
    pill = Pill(position, sys.argv[2] if len(sys.argv) > 2 else "", sys.argv[3] if len(sys.argv) > 3 else "")
    pill.show_all()
    threading.Thread(target=read_stdin, args=(pill,), daemon=True).start()
    Gtk.main()


if __name__ == "__main__":
    main()
