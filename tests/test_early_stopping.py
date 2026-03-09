"""Tests for early stopping monitor."""

import pytest

from lmt.training.early_stopping import EarlyStopping


class TestEarlyStopping:
    """Test EarlyStopping class."""

    def test_no_stop_while_improving(self):
        """Should not stop when metric keeps improving."""
        es = EarlyStopping(patience=3)
        assert not es.step(5.0)
        assert not es.step(4.0)
        assert not es.step(3.0)
        assert not es.step(2.0)

    def test_stops_after_patience(self):
        """Should stop after patience steps without improvement."""
        es = EarlyStopping(patience=2)
        assert not es.step(3.0)
        assert not es.step(4.0)  # worse, counter=1
        assert es.step(5.0)  # worse, counter=2 → patience exhausted

    def test_patience_resets_on_improvement(self):
        """Should reset patience counter when metric improves."""
        es = EarlyStopping(patience=2)
        assert not es.step(5.0)
        assert not es.step(6.0)  # worse, counter=1
        assert not es.step(4.0)  # improves! counter=0
        assert not es.step(5.0)  # worse, counter=1
        assert es.step(6.0)  # worse, counter=2 → stop

    def test_higher_is_better(self):
        """Should track higher-is-better metrics like accuracy."""
        es = EarlyStopping(patience=2, higher_is_better=True)
        assert not es.step(0.5)
        assert not es.step(0.6)  # improving
        assert not es.step(0.55)  # worse, counter=1
        assert es.step(0.58)  # still worse than best, counter=2 → stop

    def test_min_delta(self):
        """Should require minimum improvement to count."""
        es = EarlyStopping(patience=2, min_delta=0.1)
        assert not es.step(5.0)
        assert not es.step(4.95)  # only 0.05 improvement, counter=1
        assert es.step(4.90)  # still not enough, counter=2 → stop

    def test_min_delta_with_real_improvement(self):
        """Should reset when improvement exceeds min_delta."""
        es = EarlyStopping(patience=2, min_delta=0.1)
        assert not es.step(5.0)
        assert not es.step(4.95)  # not enough, counter=1
        assert not es.step(4.5)  # real improvement! counter=0
        assert not es.step(4.45)  # not enough, counter=1
        assert es.step(4.42)  # not enough, counter=2 → stop

    def test_best_value_tracked(self):
        """Should track the best metric value seen."""
        es = EarlyStopping(patience=5)
        es.step(5.0)
        es.step(3.0)
        es.step(4.0)
        assert es.best_value == 3.0

    def test_best_value_higher_is_better(self):
        """Should track best for higher-is-better."""
        es = EarlyStopping(patience=5, higher_is_better=True)
        es.step(0.5)
        es.step(0.8)
        es.step(0.6)
        assert es.best_value == 0.8

    def test_counter_property(self):
        """Should expose the current patience counter."""
        es = EarlyStopping(patience=3)
        es.step(5.0)
        assert es.counter == 0
        es.step(6.0)  # worse
        assert es.counter == 1
        es.step(7.0)  # worse
        assert es.counter == 2

    def test_patience_zero_raises(self):
        """Should reject zero patience."""
        with pytest.raises(ValueError, match='patience'):
            EarlyStopping(patience=0)

    def test_negative_min_delta_raises(self):
        """Should reject negative min_delta."""
        with pytest.raises(ValueError, match='min_delta'):
            EarlyStopping(patience=3, min_delta=-0.1)
