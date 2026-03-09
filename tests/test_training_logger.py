"""Tests for training metrics logger."""

from lmt.training.logger import TrainingLogger


class TestTrainingLogger:
    """Test training metrics logging."""

    def test_log_scalar(self):
        """Should log scalar metrics with step numbers."""
        logger = TrainingLogger()
        logger.log('loss', 2.5, step=0)
        logger.log('loss', 2.0, step=1)

        history = logger.get_history('loss')
        assert len(history) == 2
        assert history[0] == (0, 2.5)

    def test_log_dict(self):
        """Should log multiple metrics at once."""
        logger = TrainingLogger()
        logger.log_dict({'loss': 2.5, 'lr': 1e-4}, step=0)

        assert logger.get_history('loss') == [(0, 2.5)]
        assert logger.get_history('lr') == [(0, 1e-4)]

    def test_latest_value(self):
        """Should return the latest value for a metric."""
        logger = TrainingLogger()
        logger.log('loss', 3.0, step=0)
        logger.log('loss', 2.0, step=1)
        logger.log('loss', 1.0, step=2)

        assert logger.latest('loss') == 1.0

    def test_latest_missing_metric(self):
        """Should return None for unknown metrics."""
        logger = TrainingLogger()
        assert logger.latest('nonexistent') is None

    def test_summary(self):
        """Should produce a summary string."""
        logger = TrainingLogger()
        logger.log('loss', 2.5, step=0)
        logger.log('bpb', 1.5, step=0)

        summary = logger.summary()
        assert 'loss' in summary
        assert 'bpb' in summary

    def test_step_summary(self):
        """Should format a single step's metrics."""
        logger = TrainingLogger()
        logger.log('loss', 2.5, step=10)
        logger.log('lr', 1e-4, step=10)

        line = logger.step_summary(step=10)
        assert 'step 10' in line.lower() or '10' in line
        assert 'loss' in line

    def test_reset(self):
        """Should clear all logged metrics."""
        logger = TrainingLogger()
        logger.log('loss', 2.5, step=0)
        logger.reset()
        assert logger.get_history('loss') == []

    def test_metric_names(self):
        """Should list all logged metric names."""
        logger = TrainingLogger()
        logger.log('loss', 2.5, step=0)
        logger.log('bpb', 1.5, step=0)
        logger.log('accuracy', 0.5, step=0)

        names = logger.metric_names()
        assert set(names) == {'loss', 'bpb', 'accuracy'}

    def test_to_dict(self):
        """Should serialize all history to dict."""
        logger = TrainingLogger()
        logger.log('loss', 2.5, step=0)
        logger.log('loss', 2.0, step=1)

        d = logger.to_dict()
        assert 'loss' in d
        assert len(d['loss']) == 2
