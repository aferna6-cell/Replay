"""Tests for KL schedule and Experiment 6 machinery."""

import pytest

from ml.kl_schedule import (EXPERIMENT_6_KL_SCHEDULE, parse_kl_schedule,
                            schedule_table)


def test_experiment_6_schedule_hold_phase():
    coef = parse_kl_schedule(EXPERIMENT_6_KL_SCHEDULE)
    assert coef(1) == pytest.approx(0.03)
    assert coef(160) == pytest.approx(0.03)


def test_experiment_6_schedule_anneal_start():
    coef = parse_kl_schedule(EXPERIMENT_6_KL_SCHEDULE)
    assert coef(161) == pytest.approx(0.03)


def test_experiment_6_schedule_anneal_end():
    coef = parse_kl_schedule(EXPERIMENT_6_KL_SCHEDULE)
    assert coef(320) == pytest.approx(0.01)


def test_experiment_6_schedule_midpoint():
    coef = parse_kl_schedule(EXPERIMENT_6_KL_SCHEDULE)
    mid = coef(240)
    assert 0.01 < mid < 0.03


def test_schedule_table_checkpoints():
    table = schedule_table(EXPERIMENT_6_KL_SCHEDULE, (0, 160, 161, 320))
    by_iter = {r["iteration"]: r["kl_coef"] for r in table}
    assert by_iter[160] == pytest.approx(0.03)
    assert by_iter[161] == pytest.approx(0.03)
    assert by_iter[320] == pytest.approx(0.01)


def test_invalid_schedule_rejected():
    with pytest.raises(ValueError):
        parse_kl_schedule("0.03")
