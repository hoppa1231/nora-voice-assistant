"""On-screen status pill: listening / thinking / speaking.

Runs as a separate process under the *system* Python (GTK and gtk-layer-shell come
from the distro, not from the venv) and reads JSON lines on stdin:

    {"state": "attentive", "seconds": 8, "badge": "Клод работает"}

It exits when stdin closes, i.e. together with the assistant. Mouse clicks pass
through it. Standalone on purpose: no imports from the package.
"""

from __future__ import annotations

import json
import math
import sys
import threading
import time

import cairo
import gi

gi.require_version("Gtk", "3.0")
gi.require_version("GtkLayerShell", "0.1")
gi.require_version("PangoCairo", "1.0")
from gi.repository import GLib, Gtk, GtkLayerShell, Pango, PangoCairo  # noqa: E402

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
    def __init__(self, position: str) -> None:
        super().__init__()
        self.state = "idle"
        self.badge = ""
        self.until = 0.0  # attentive countdown end (monotonic)
        self.total = 0.0
        self.changed = time.monotonic()
        self.width = HEIGHT

        GtkLayerShell.init_for_window(self)
        GtkLayerShell.set_layer(self, GtkLayerShell.Layer.OVERLAY)
        GtkLayerShell.set_namespace(self, "nora-indicator")
        edge = GtkLayerShell.Edge.TOP if position == "top" else GtkLayerShell.Edge.BOTTOM
        GtkLayerShell.set_anchor(self, edge, True)
        GtkLayerShell.set_margin(self, edge, 56 if position == "top" else 26)
        GtkLayerShell.set_keyboard_mode(self, GtkLayerShell.KeyboardMode.NONE)
        GtkLayerShell.set_exclusive_zone(self, -1)

        visual = self.get_screen().get_rgba_visual()
        if visual:
            self.set_visual(visual)
        self.set_app_paintable(True)
        self.area = Gtk.DrawingArea()
        self.area.connect("draw", self.draw)
        self.add(self.area)
        self.connect("realize", lambda *_: self.get_window().input_shape_combine_region(cairo.Region(), 0, 0))
        self.resize_to_content()
        GLib.timeout_add(33, self.tick)

    def update(self, message: dict) -> bool:
        state = message.get("state", self.state)
        if state != self.state:
            self.changed = time.monotonic()
        self.state = state if state in STATES else "idle"
        self.badge = message.get("badge", "")
        seconds = float(message.get("seconds") or 0)
        if seconds:
            self.until, self.total = time.monotonic() + seconds, seconds
        self.resize_to_content()
        return False

    def label(self) -> str:
        text = STATES[self.state][1]
        return " · ".join(t for t in (text, self.badge) if t)

    def resize_to_content(self) -> None:
        text = self.label()
        width = HEIGHT
        if text:
            layout = self.create_pango_layout(text)
            layout.set_font_description(FONT)
            width = PAD + DOT * 2 + 10 + layout.get_pixel_size()[0] + PAD + 2
        if width != self.width or not self.get_visible():
            self.width = width
            self.area.set_size_request(width, HEIGHT)
            self.resize(width, HEIGHT)

    def tick(self) -> bool:
        self.area.queue_draw()
        return True

    def draw(self, _widget, cr: cairo.Context) -> None:
        now = time.monotonic()
        t = now - self.changed
        w, h = self.width, HEIGHT
        cr.set_operator(cairo.OPERATOR_SOURCE)
        cr.set_source_rgba(0, 0, 0, 0)
        cr.paint()
        cr.set_operator(cairo.OPERATOR_OVER)

        quiet = self.state in ("idle", "hearing") and not self.badge
        r = h / 2
        cr.new_sub_path()
        cr.arc(r, r, r, math.pi / 2, 3 * math.pi / 2)
        cr.arc(w - r, r, r, 3 * math.pi / 2, math.pi / 2)
        cr.close_path()
        cr.set_source_rgba(0.08, 0.09, 0.11, 0.45 if quiet else 0.82)
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
            cr.move_to(PAD + DOT * 2 + 10, (h - th) / 2)
            PangoCairo.show_layout(cr, layout)

        if self.state == "attentive" and self.until > now and self.total:  # follow-up countdown
            left = (self.until - now) / self.total
            cr.set_source_rgba(cr_, cg, cb, 0.8)
            cr.rectangle(r, h - 3, (w - 2 * r) * left, 2)
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
    pill = Pill(position)
    pill.show_all()
    threading.Thread(target=read_stdin, args=(pill,), daemon=True).start()
    Gtk.main()


if __name__ == "__main__":
    main()
