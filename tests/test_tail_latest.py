import threading

from hsbg_coach.tail import tail_latest


def test_tail_latest_switches_to_new_session(tmp_path):
    first = tmp_path / "old.log"
    second = tmp_path / "new.log"
    first.write_text("old line\n", encoding="utf-8")
    current = [str(first)]
    stop = threading.Event()
    stream = tail_latest(lambda: current[0], from_start=True,
                         poll_interval=0.001, stop_event=stop)
    assert next(stream) == "old line"

    second.write_text("new line\n", encoding="utf-8")
    current[0] = str(second)
    assert next(stream) == "new line"
    stop.set()
