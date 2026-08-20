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

    note = snapshot.get("build_note")
    if note:
        lines.append(note)

    if recommendations:
        # Lead with ONE clear next move; it re-computes the instant you act, so
        # do this, then read the new top line. The rest is context below it.
        lines.append("")
        lines.append(f"NEXT → {recommendations[0]}")
        if len(recommendations) > 1:
            lines.append("")
            lines.append("then:")
            for r in recommendations[1:]:
                lines.append(f"  · {r}")

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


def format_next(snapshot: Dict, odds: Optional[str] = None,
                recommendations: Optional[List[str]] = None) -> str:
    """Minimal one-move view: the single best NEXT action + a compact status line.
    No board/shop dump — just 'what to do now', which re-computes as you act."""
    turn = snapshot.get("turn")
    tier = snapshot.get("tavern_tier")
    gold = snapshot.get("gold")
    hp = snapshot.get("hero_health")
    phase = snapshot.get("phase", "?")
    status = f"turn {turn} · {phase} · tier {tier} · gold {gold} · hp {hp}"
    hpw = snapshot.get("hero_power")
    if hpw and hpw.get("usable"):
        status += " · hero power ready"
    if snapshot.get("anomaly"):
        status += f" · anomaly: {snapshot['anomaly']}"
    # Sync counter — ticks up every time a new board/shop is ingested, so after a
    # roll you can see the panel re-read the tavern (it's processing, not stuck).
    seq = snapshot.get("sync_seq")
    if seq is not None:
        status += f" · synced ✓ #{seq}"

    if recommendations:
        # Lead with the single best move, then list a couple of alternatives below
        # it (smaller) so you can override the top pick when you disagree.
        out = [f"→ {recommendations[0]}", f"  {status}"]
        # Alternatives that aren't just the top line again (duplicate shop
        # entities produced identical lines — live 2026-08-20).
        seen = {recommendations[0]}
        alts = [a for a in recommendations[1:] if a not in seen and not seen.add(a)]
        if alts[:2]:
            out.append("  or:")
            out.extend(f"   - {a}" for a in alts[:2])
        return "\n".join(out)
    # No move to make right now (combat / hero-select / between turns): just show
    # the status line, no combat screen.
    return f"  {status}"


class Overlay:
    """Always-on-top text panel that floats over a windowed game.

    Frameless (no title bar) mode is sleek but macOS's Tk hides
    ``overrideredirect`` windows, so it's opt-in and OFF by default on macOS —
    there we use a normal top-most window that reliably renders.

    Usage::

        ov = Overlay()
        ov.update(snapshot_dict, recommendations=[...])
        ov.poll(provider, interval_ms=500)   # provider() -> (snapshot, odds, recos)
        ov.run()
    """

    def __init__(self, corner: str = "ne", alpha: Optional[float] = None,
                 width: int = 320, height: int = 460,
                 frameless: Optional[bool] = None) -> None:
        import sys
        import tkinter as tk  # lazy: keeps module headless-importable

        is_mac = sys.platform == "darwin"
        # Frameless hides the window on macOS — default it off there.
        if frameless is None:
            frameless = not is_mac
        # Window transparency stops the text layer from compositing on some macOS
        # Tk builds (the panel paints its background but not its glyphs), so the
        # panel looks blank. Default to fully opaque on macOS; keep a little
        # translucency elsewhere where it renders fine.
        if alpha is None:
            alpha = 1.0 if is_mac else 0.92

        self.root = tk.Tk()
        self.root.title("HSBG Coach")
        if frameless:
            try:
                self.root.overrideredirect(True)      # no title bar / border
            except Exception:
                pass
        self._place(width, height, corner)

        # Render into a Text widget, not a Label. Apple's deprecated *system* Tk
        # (8.5) has a long-standing bug where a Label paints its background but
        # not its glyphs, so the panel looks blank — a Text widget paints
        # reliably on the same build. (Heads-up so the user can upgrade Tk.)
        tkver = float(self.root.tk.call("info", "patchlevel").rsplit(".", 1)[0]) \
            if "." in self.root.tk.call("info", "patchlevel") else 0.0
        if is_mac and tkver and tkver < 8.6:
            print("Note: macOS system Tk", self.root.tk.call("info", "patchlevel"),
                  "is old; using a Text-based panel. For a crisper overlay install"
                  " Tk 8.6 (brew install python-tk).")

        self._text = tk.Text(
            self.root, wrap="word", relief="flat", borderwidth=0,
            font=("Menlo", 12), fg="#e8e8e8", bg="#111418",
            padx=10, pady=10, highlightthickness=0, cursor="arrow",
        )
        self._text.insert("1.0", "HSBG Coach — waiting for Hearthstone…")
        self._text.config(state="disabled")
        self._text.pack(fill="both", expand=True)
        self._text.bind("<Button-1>", self._start_drag)
        self._text.bind("<B1-Motion>", self._on_drag)
        self._drag = (0, 0)

        # Force the window visible and on top (macOS especially needs the kick).
        try:
            self.root.attributes("-alpha", alpha)
        except tk.TclError:
            pass
        self.root.update_idletasks()
        self.root.deiconify()
        self.root.lift()
        self.root.attributes("-topmost", True)
        self._keep_on_top()

    def _keep_on_top(self) -> None:
        # Re-assert topmost so it keeps floating over the game (without stealing
        # focus — we set the attribute, we don't raise/focus the window).
        try:
            self.root.attributes("-topmost", True)
        except Exception:
            pass
        self.root.after(2000, self._keep_on_top)

    def _place(self, w: int, h: int, corner: str) -> None:
        sw = self.root.winfo_screenwidth()
        x = sw - w - 20 if "e" in corner else 20
        y = 40 if "n" in corner else 20
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    def _start_drag(self, e) -> None:
        self._drag = (e.x, e.y)

    def _on_drag(self, e) -> None:
        x = self.root.winfo_x() + (e.x - self._drag[0])
        y = self.root.winfo_y() + (e.y - self._drag[1])
        self.root.geometry(f"+{x}+{y}")

    def update(self, snapshot: Dict, odds: Optional[str] = None,
               recommendations: Optional[List[str]] = None) -> None:
        text = format_overlay_text(snapshot, odds, recommendations)
        self._text.config(state="normal")
        self._text.delete("1.0", "end")
        self._text.insert("1.0", text)
        self._text.config(state="disabled")
        # Force the pending redraw so updates from after() callbacks actually show.
        try:
            self._text.update_idletasks()
        except Exception:
            pass

    def poll(self, provider: Callable[[], Optional[tuple]], interval_ms: int = 500) -> None:
        """Call provider() every interval; it returns (snapshot, odds[, recos]) or None."""
        def tick():
            try:
                result = provider()
                if result:
                    self.update(*result)         # (snapshot, odds) or (snapshot, odds, recos)
            except Exception as exc:  # never let a bad frame kill the overlay
                try:
                    self._text.config(state="normal")
                    self._text.delete("1.0", "end")
                    self._text.insert("1.0", f"overlay error: {exc}")
                    self._text.config(state="disabled")
                except Exception:
                    pass
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
