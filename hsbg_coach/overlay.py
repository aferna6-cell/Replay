"""Transparent always-on-top overlay window (Tkinter, stdlib — no deps).

Renders the current Battlegrounds snapshot plus combat odds in a small frameless
window pinned to a screen corner, like HDT's overlays. The text rendering is
split into a pure function (``format_overlay_text``) so it is unit-testable
without a display; the Tk class is a thin shell around it.

Tkinter is imported lazily inside the GUI code so this module imports cleanly in
a headless environment (CI, servers) where only the pure formatter is exercised.
"""

from typing import Callable, Dict, List, Optional


def _fmt_minion(m: Dict) -> str:
    name = m.get("name") or m.get("card_id") or "?"
    atk, hp = m.get("attack"), m.get("health")
    stats = f" {atk}/{hp}" if atk is not None or hp is not None else ""
    pos = m.get("position")
    prefix = f"{pos}. " if pos is not None else "- "
    return f"  {prefix}{name}{stats}"


def format_overlay_text(snapshot: Dict, odds: Optional[str] = None,
                        recommendations: Optional[List[str]] = None) -> str:
    """Pure renderer: snapshot dict (Snapshot.to_dict()) -> display text.

    `recommendations` is the advisor's ranked move list (best first); it's shown
    prominently near the top — that's the whole point of the overlay.
    """
    lines: List[str] = []
    turn = snapshot.get("turn")
    tier = snapshot.get("tavern_tier")
    gold = snapshot.get("gold")
    hp = snapshot.get("hero_health")
    phase = snapshot.get("phase", "?")
    lines.append(f"HSBG Coach — turn {turn}  ·  {phase}")
    lines.append(f"tier {tier}   gold {gold}   hp {hp}")

    if recommendations:
        lines.append("")
        lines.append("▸ Recommended:")
        for i, r in enumerate(recommendations, 1):
            lines.append(f"  {i}. {r}")

    if odds:
        lines.append("")
        lines.append(f"Combat: {odds}")

    board = snapshot.get("board") or []
    lines.append("")
    lines.append(f"Your board ({len(board)}):")
    lines.extend(_fmt_minion(m) for m in board) if board else lines.append("  (empty)")

    shop = snapshot.get("shop") or []
    if shop:
        lines.append("")
        lines.append(f"Shop ({len(shop)}):")
        lines.extend(_fmt_minion(m) for m in shop)

    for note in snapshot.get("notes") or []:
        lines.append(f"  · {note}")

    return "\n".join(lines)


class Overlay:
    """Frameless, always-on-top, semi-transparent text panel.

    Usage::

        ov = Overlay()
        ov.update(snapshot_dict, odds="win 62% / tie 8% / loss 30%")
        ov.poll(provider, interval_ms=500)   # provider() -> (snapshot, odds)
        ov.run()
    """

    def __init__(self, corner: str = "ne", alpha: float = 0.85,
                 width: int = 300, height: int = 420) -> None:
        import tkinter as tk  # lazy: keeps module headless-importable

        self.root = tk.Tk()
        self.root.overrideredirect(True)          # no title bar / border
        self.root.attributes("-topmost", True)    # always on top
        try:
            self.root.attributes("-alpha", alpha)  # window transparency
        except tk.TclError:
            pass  # some platforms/WMs don't support per-window alpha
        self._place(width, height, corner)

        self._label = tk.Label(
            self.root, justify="left", anchor="nw",
            font=("Menlo", 12), fg="#e8e8e8", bg="#111418",
            padx=10, pady=10, text="HSBG Coach — waiting for a game…",
        )
        self._label.pack(fill="both", expand=True)
        # Drag-to-move (frameless windows can't be moved otherwise).
        self._label.bind("<Button-1>", self._start_drag)
        self._label.bind("<B1-Motion>", self._on_drag)
        self._drag = (0, 0)

    def _place(self, w: int, h: int, corner: str) -> None:
        sw = self.root.winfo_screenwidth()
        x = sw - w - 20 if "e" in corner else 20
        y = 20 if "n" in corner else 20
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    def _start_drag(self, e) -> None:
        self._drag = (e.x, e.y)

    def _on_drag(self, e) -> None:
        x = self.root.winfo_x() + (e.x - self._drag[0])
        y = self.root.winfo_y() + (e.y - self._drag[1])
        self.root.geometry(f"+{x}+{y}")

    def update(self, snapshot: Dict, odds: Optional[str] = None,
               recommendations: Optional[List[str]] = None) -> None:
        self._label.config(text=format_overlay_text(snapshot, odds, recommendations))

    def poll(self, provider: Callable[[], Optional[tuple]], interval_ms: int = 500) -> None:
        """Call provider() every interval; it returns (snapshot, odds[, recos]) or None."""
        def tick():
            try:
                result = provider()
                if result:
                    self.update(*result)         # (snapshot, odds) or (snapshot, odds, recos)
            except Exception as exc:  # never let a bad frame kill the overlay
                self._label.config(text=f"overlay error: {exc}")
            self.root.after(interval_ms, tick)
        self.root.after(interval_ms, tick)

    def run(self) -> None:
        self.root.mainloop()


def demo() -> None:
    """Show the overlay with a static sample snapshot (needs a display)."""
    sample = {
        "turn": 5, "phase": "recruit", "tavern_tier": 2, "gold": 8,
        "hero_health": 27,
        "board": [
            {"position": 1, "name": "Tabbycat", "attack": 1, "health": 1},
            {"position": 2, "name": "Alleycat", "attack": 1, "health": 1},
        ],
        "shop": [
            {"name": "Rockpool Hunter", "attack": 2, "health": 3},
            {"name": "Murloc Tidehunter", "attack": 2, "health": 1},
        ],
        "notes": [],
    }
    ov = Overlay()
    ov.update(sample, odds="win 62% / tie 8% / loss 30%")
    ov.run()


if __name__ == "__main__":
    demo()
