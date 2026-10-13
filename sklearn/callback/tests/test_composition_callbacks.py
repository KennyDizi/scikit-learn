# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

"""Callback behavior of the parallel composition meta-estimators."""

import warnings

import numpy as np
import pytest
from joblib import parallel_backend

from sklearn import config_context
from sklearn.base import (
    BaseEstimator,
    RegressorMixin,
    TransformerMixin,
    _fit_context,
    clone,
)
from sklearn.callback import CallbackSupportMixin, ProgressBar, ScoringMonitor
from sklearn.callback.tests._common.callbacks import (
    RecordingAutoPropagatedCallback,
    RecordingCallback,
    StopFitCallback,
)
from sklearn.callback.tests._common.estimators import MaxIterEstimator
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_regression
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.linear_model import LogisticRegression, Ridge, SGDRegressor
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import assert_allclose, assert_array_equal

BACKENDS = ["sequential", "threading", "loky"]


class IterRegressor(RegressorMixin, MaxIterEstimator):
    """MaxIterEstimator tagged as a regressor so voting can wrap it."""


class RaisingCallbackEstimator(CallbackSupportMixin, BaseEstimator):
    _parameter_constraints: dict = {}

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        callback_ctx = self._init_callback_context()
        callback_ctx.call_on_fit_task_begin(estimator=self, X=X, y=y)
        raise ValueError("boom")

    def transform(self, X):
        return X


class RoutingRegressor(RegressorMixin, BaseEstimator):
    _parameter_constraints: dict = {}

    def fit(self, X, y, sample_weight=None, metadata=None):
        self.sample_weight_ = sample_weight
        self.metadata_ = metadata
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return np.zeros(X.shape[0])


class RoutingTransformer(TransformerMixin, BaseEstimator):
    _parameter_constraints: dict = {}

    def fit(self, X, y=None, sample_weight=None, metadata=None):
        self.sample_weight_ = sample_weight
        self.metadata_ = metadata
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return X


class MarkParent(RecordingCallback):
    """Sets a marker on the estimator object that invoked the hook."""

    def on_fit_task_begin(self, estimator, context):
        super().on_fit_task_begin(estimator, context)
        if context.parent is not None:
            estimator._worker_marker = "from-worker"


class _RootSnapshot(RecordingCallback):
    def __init__(self):
        super().__init__()
        self.root_ends = []

    def on_fit_task_end(
        self,
        estimator,
        context,
        *,
        X=None,
        y=None,
        metadata=None,
        fitted_estimator=None,
    ):
        super().on_fit_task_end(
            estimator,
            context,
            X=X,
            y=y,
            metadata=metadata,
            fitted_estimator=fitted_estimator,
        )
        if context.parent is None:
            self.root_ends.append(estimator)


def _iter_regressor(max_iter=2):
    return IterRegressor(max_iter=max_iter, computation_intensity=0)


def _root_records(callback, hook_name):
    return [
        record
        for record in callback.record
        if record["name"] == hook_name and record["context"].parent is None
    ]


def _task_names(callback, hook_name="on_fit_task_begin"):
    return [
        record["context"].task_name
        for record in callback.record
        if record["name"] == hook_name
    ]


def _assert_clean(estimator):
    assert not hasattr(estimator, "_callback_fit_ctx")
    assert not hasattr(estimator, "_skl_callbacks_to_teardown")
    assert not hasattr(estimator, "_parent_callback_ctx")


def _nested_voting():
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("est", _iter_regressor()),
        ]
    )
    return VotingRegressor(
        [("pipe", pipeline), ("direct", _iter_regressor())],
        n_jobs=2,
    )


def _nested_multioutput():
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("est", _iter_regressor()),
        ]
    )
    return MultiOutputRegressor(pipeline, n_jobs=2)


def _nested_union():
    columns = ColumnTransformer(
        [
            ("scale", StandardScaler(), [0, 1]),
            ("pass", "passthrough", [2, 3]),
        ],
        n_jobs=2,
    )
    return FeatureUnion(
        [
            ("columns", columns),
            ("scale", StandardScaler()),
            ("drop", "drop"),
        ],
        n_jobs=2,
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_nested_repeated_fits(backend):
    """Nested compositions propagate callbacks on every parallel backend.

    A second fit appends another full run. Temporary callback state is gone
    afterwards, and repeated predictions stay the same.
    """
    X, y = make_regression(n_samples=20, n_features=4, random_state=0)
    _, y_multi = make_regression(
        n_samples=20, n_features=4, n_targets=2, random_state=0
    )

    cases = [
        (_nested_voting(), X, y, "predict"),
        (_nested_multioutput(), X, y_multi, "predict"),
        (_nested_union(), X, None, "transform"),
    ]
    for estimator, x_case, y_case, method_name in cases:
        callback = RecordingAutoPropagatedCallback()
        callback.max_propagation_depth = None
        estimator.set_callbacks(callback)
        fit_args = (x_case,) if y_case is None else (x_case, y_case)
        with parallel_backend(backend):
            estimator.fit(*fit_args)
            first = getattr(estimator, method_name)(x_case)
            begin_once = callback.count_hooks("on_fit_task_begin")
            end_once = callback.count_hooks("on_fit_task_end")
            assert callback.count_hooks("setup") == 1
            assert callback.count_hooks("teardown") == 1
            assert begin_once == end_once
            assert begin_once > 1
            estimator.fit(*fit_args)
            second = getattr(estimator, method_name)(x_case)
        assert callback.count_hooks("setup") == 2
        assert callback.count_hooks("teardown") == 2
        assert callback.count_hooks("on_fit_task_begin") == 2 * begin_once
        assert callback.count_hooks("on_fit_task_end") == 2 * end_once
        assert_allclose(first, second)
        _assert_clean(estimator)


@pytest.mark.parametrize("backend", BACKENDS)
def test_propagation_depth_and_child_callbacks(backend):
    """Depth limits which nested estimators see an auto-propagated callback.

    A non-propagated callback registered on a child still runs, and a child
    that already has an auto-propagated callback is rejected.
    """
    X, y = make_regression(n_samples=16, n_features=3, random_state=0)

    def fit_voting(depth):
        callback = RecordingAutoPropagatedCallback()
        callback.max_propagation_depth = depth
        estimator = _nested_voting().set_callbacks(callback)
        with parallel_backend(backend):
            estimator.fit(X, y)
        return callback

    shallow = fit_voting(0)
    middle = fit_voting(1)
    full = fit_voting(None)

    # depth 0: voting root + 2 children. Inner iteration tasks stay hidden.
    assert shallow.count_hooks("setup") == 1
    assert shallow.count_hooks("teardown") == 1
    assert shallow.count_hooks("on_fit_task_begin") == 3
    assert not any(name.startswith("iteration") for name in _task_names(shallow))

    # depth 1 reaches both direct children. The pipeline does not propagate
    # further, so the inner regressor's iterations are hidden. The direct
    # regressor still reports its two iterations: 7 tasks in total.
    assert middle.count_hooks("on_fit_task_begin") == 7
    assert _task_names(middle).count("iteration 0") == 1

    # Unlimited depth also includes the inner regressor's iterations.
    assert _task_names(full).count("iteration 0") == 2
    assert full.count_hooks("on_fit_task_begin") == 9
    assert full.count_hooks("on_fit_task_begin") == full.count_hooks("on_fit_task_end")

    child_callback = RecordingCallback()
    voting = VotingRegressor(
        [("est", _iter_regressor().set_callbacks(child_callback))],
        n_jobs=2,
    )
    parent_callback = RecordingAutoPropagatedCallback()
    with parallel_backend(backend):
        voting.set_callbacks(parent_callback).fit(X, y)
    assert child_callback.count_hooks("setup") == 1
    assert child_callback.count_hooks("teardown") == 1
    # Child root plus two iterations. The parent callback is not in this count.
    assert child_callback.count_hooks("on_fit_task_begin") == 3
    assert child_callback.count_hooks("on_fit_task_end") == 3
    stored = voting.estimators_[0]
    template = voting.estimators[0][1]
    # loky returns an unpickled copy of the child callback. The instance that
    # was registered on the template is the one that received the events.
    assert template._skl_callbacks == [child_callback]
    assert not hasattr(template, "_parent_callback_ctx")
    assert not hasattr(stored, "_parent_callback_ctx")
    assert not any(
        isinstance(callback, RecordingAutoPropagatedCallback)
        for callback in getattr(stored, "_skl_callbacks", [])
    )

    conflicting = _iter_regressor().set_callbacks(RecordingAutoPropagatedCallback())
    voting = VotingRegressor([("est", conflicting)], n_jobs=2)
    parent_callback = RecordingAutoPropagatedCallback()
    voting.set_callbacks(parent_callback)
    with parallel_backend(backend):
        with pytest.raises(TypeError, match="auto-propagated callbacks"):
            voting.fit(X, y)
    assert parent_callback.count_hooks("setup") == 1
    assert parent_callback.count_hooks("teardown") == 1
    assert _root_records(parent_callback, "on_fit_task_end") == []
    _assert_clean(voting)
    _assert_clean(conflicting)
    assert len(conflicting._skl_callbacks) == 1


@pytest.mark.parametrize("backend", BACKENDS)
def test_stop_does_not_cancel_dispatched_siblings(backend):
    """A stop request does not drop siblings or change predictions."""
    X, y = make_regression(n_samples=20, n_features=3, random_state=0)
    estimators = [
        ("a", Ridge()),
        ("b", Ridge(alpha=0.5)),
        ("c", Ridge(alpha=2.0)),
    ]
    baseline = VotingRegressor(estimators, n_jobs=2).fit(X, y)
    watched = VotingRegressor(estimators, n_jobs=2).set_callbacks(StopFitCallback())
    with parallel_backend(backend):
        watched.fit(X, y)
    assert len(watched.estimators_) == 3
    assert_allclose(baseline.predict(X), watched.predict(X))
    assert not hasattr(watched, "_worker_marker")
    _assert_clean(watched)


@pytest.mark.parametrize("backend", BACKENDS)
def test_worker_local_marker_is_not_merged(backend):
    """Mutations of the pickled parent inside a worker stay in that worker."""
    X, y = make_regression(n_samples=12, n_features=2, random_state=0)
    estimator = VotingRegressor(
        [("a", Ridge()), ("b", Ridge())],
        n_jobs=2,
    ).set_callbacks(MarkParent())
    with parallel_backend(backend):
        estimator.fit(X, y)
    if backend == "loky":
        # The worker mutated a copy. That copy is not merged into the parent.
        assert not hasattr(estimator, "_worker_marker")
    else:
        # Sequential and threading run the hook on the live parent.
        assert estimator._worker_marker == "from-worker"
    for child in estimator.estimators_:
        assert not hasattr(child, "_parent_callback_ctx")
        assert not hasattr(child, "_worker_marker")


@pytest.mark.parametrize("backend", BACKENDS)
def test_estimator_error_cleans_up_without_success(backend):
    """A failing child does not report a successful root fit or leave parent state."""
    X = np.arange(12, dtype=float).reshape(6, 2)
    child_callback = RecordingCallback()
    failing = RaisingCallbackEstimator().set_callbacks(child_callback)
    reusable = StandardScaler()
    union = FeatureUnion(
        [("ok", reusable), ("bad", failing)],
        n_jobs=2,
    )
    parent_callback = RecordingAutoPropagatedCallback()
    union.set_callbacks(parent_callback)

    with parallel_backend(backend):
        with pytest.raises(ValueError, match="boom"):
            union.fit(X)

    assert parent_callback.count_hooks("setup") == 1
    assert parent_callback.count_hooks("teardown") == 1
    assert _root_records(parent_callback, "on_fit_task_end") == []
    assert child_callback.count_hooks("setup") == 1
    assert child_callback.count_hooks("teardown") == 1
    assert child_callback.count_hooks("on_fit_task_begin") == 1
    assert child_callback.count_hooks("on_fit_task_end") == 0
    _assert_clean(union)
    _assert_clean(failing)
    _assert_clean(reusable)
    assert failing._skl_callbacks == [child_callback]

    union.transformer_list = [
        ("ok", StandardScaler()),
        ("also", StandardScaler()),
    ]
    with parallel_backend(backend):
        transformed = union.fit_transform(X)
    assert transformed.shape[0] == X.shape[0]
    _assert_clean(union)


@pytest.mark.parametrize("backend", BACKENDS)
def test_builtin_listener_handles_survive_workers(backend):
    """Progress bars keep the parent's listener and close it after fit."""
    pytest.importorskip("rich")
    X, y = make_regression(n_samples=16, n_features=3, n_targets=2, random_state=0)
    progress = ProgressBar(max_propagation_depth=None)
    monitor = ScoringMonitor(scoring="r2")
    listener = monitor._listener_handle
    estimator = MultiOutputRegressor(_iter_regressor(max_iter=3), n_jobs=2)
    estimator.set_callbacks(progress, monitor)
    with parallel_backend(backend):
        estimator.fit(X, y)
    assert monitor._listener_handle is listener
    assert progress._listener_handles == {}
    _assert_clean(estimator)


def test_passthrough_does_not_warn_about_callback_support():
    """String passthrough is a real task and is not reported as unsupported."""
    X = np.arange(8, dtype=float).reshape(4, 2)
    callback = RecordingAutoPropagatedCallback()
    union = FeatureUnion(
        [("pass", "passthrough"), ("scale", StandardScaler())]
    ).set_callbacks(callback)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        transformed = union.fit_transform(X)
    assert transformed.shape == (4, 4)
    # Root, passthrough step, scaler step.
    assert callback.count_hooks("on_fit_task_begin") == 3
    assert callback.count_hooks("setup") == 1
    assert callback.count_hooks("teardown") == 1


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "builder",
    [
        lambda: VotingRegressor(
            [("a", Ridge()), ("b", Ridge(alpha=10.0)), ("c", "drop")],
            n_jobs=2,
        ),
        lambda: VotingClassifier(
            [
                ("a", LogisticRegression(max_iter=200)),
                ("b", LogisticRegression(max_iter=200, C=0.5)),
            ],
            voting="soft",
            n_jobs=2,
        ),
        lambda: MultiOutputRegressor(Ridge(), n_jobs=2),
        lambda: MultiOutputClassifier(LogisticRegression(max_iter=200), n_jobs=2),
        lambda: FeatureUnion(
            [
                ("scale", StandardScaler()),
                ("pass", "passthrough"),
            ],
            transformer_weights={"scale": 2.0},
            n_jobs=2,
        ),
        lambda: ColumnTransformer(
            [("scale", StandardScaler(), [0, 1])],
            remainder="passthrough",
            n_jobs=2,
        ),
    ],
)
def test_behavior_without_callbacks_is_preserved(backend, builder):
    """Callbacks do not change predictions, output shape, or get_params."""
    X, y = make_regression(n_samples=24, n_features=4, random_state=0)
    y_class = (y > np.median(y)).astype(int)
    y_multi = np.column_stack([y_class, 1 - y_class])

    reference = builder()
    watched = builder()
    params = watched.get_params(deep=True)
    callback = RecordingCallback()
    watched.set_callbacks(callback)
    assert watched.get_params(deep=True) == params
    cloned = clone(watched)
    assert cloned._skl_callbacks[0] is callback

    if isinstance(reference, (FeatureUnion, ColumnTransformer)):
        fit_args = (X,)
    elif isinstance(reference, MultiOutputClassifier):
        fit_args = (X, y_multi)
    elif isinstance(reference, MultiOutputRegressor):
        fit_args = (X, y_multi.astype(float))
    elif isinstance(reference, VotingClassifier):
        fit_args = (X, y_class)
    else:
        fit_args = (X, y)
    is_transformer = isinstance(reference, (FeatureUnion, ColumnTransformer))
    with parallel_backend(backend):
        reference.fit(*fit_args)
        watched.fit(*fit_args)
    if is_transformer:
        ref_out = reference.transform(X)
        out = watched.transform(X)
    elif isinstance(reference, VotingClassifier):
        ref_out = reference.predict_proba(X)
        out = watched.predict_proba(X)
        assert reference.transform(X).shape == watched.transform(X).shape
    else:
        ref_out = reference.predict(X)
        out = watched.predict(X)
    assert ref_out.shape == out.shape
    assert_allclose(ref_out, out)
    assert callback.count_hooks("setup") == 1
    assert callback.count_hooks("teardown") == 1
    _assert_clean(watched)


def test_metadata_routing_and_sample_weight_metadata():
    """Routed metadata still reaches children, and callbacks observe sample_weight."""
    X, y = make_regression(n_samples=10, n_features=3, n_targets=2, random_state=0)
    sample_weight = np.linspace(0.1, 1.0, num=X.shape[0])
    metadata = np.arange(X.shape[0])

    callback = RecordingCallback()
    with config_context(enable_metadata_routing=True):
        plain = RoutingRegressor().set_fit_request(sample_weight=True, metadata=True)
        with_callback = clone(plain)
        multi_plain = MultiOutputRegressor(plain, n_jobs=2)
        multi_callback = MultiOutputRegressor(with_callback, n_jobs=2).set_callbacks(
            callback
        )
        multi_plain.fit(X, y, sample_weight=sample_weight, metadata=metadata)
        multi_callback.fit(X, y, sample_weight=sample_weight, metadata=metadata)

    for fitted in (multi_plain, multi_callback):
        for estimator in fitted.estimators_:
            assert_allclose(estimator.sample_weight_, sample_weight)
            assert_array_equal(estimator.metadata_, metadata)

    root_begin = _root_records(callback, "on_fit_task_begin")
    assert len(root_begin) == 1
    assert_allclose(root_begin[0]["kwargs"]["metadata"]["sample_weight"], sample_weight)

    with config_context(enable_metadata_routing=True):
        transformer = RoutingTransformer().set_fit_request(
            sample_weight=True, metadata=True
        )
        union = FeatureUnion([("route", transformer)], n_jobs=1)
        union.set_callbacks(RecordingCallback()).fit(
            X, sample_weight=sample_weight, metadata=metadata
        )
    fitted_transformer = union.transformer_list[0][1]
    assert_allclose(fitted_transformer.sample_weight_, sample_weight)
    assert_array_equal(fitted_transformer.metadata_, metadata)


@pytest.mark.parametrize("backend", BACKENDS)
def test_classifier_attributes_are_set_before_root_end(backend):
    """``classes_`` and ``le_`` exist when the top-level fit task ends."""
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    y = np.array([0, 0, 1, 1])
    y_multi = np.column_stack([y, 1 - y])

    voting_callback = _RootSnapshot()
    voting = VotingClassifier(
        [
            ("a", LogisticRegression(max_iter=200)),
            ("b", LogisticRegression(max_iter=200)),
        ],
        n_jobs=2,
    ).set_callbacks(voting_callback)
    with parallel_backend(backend):
        voting.fit(X, y)
    assert len(voting_callback.root_ends) == 1
    seen = voting_callback.root_ends[0]
    assert_array_equal(seen.classes_, np.array([0, 1]))
    assert_array_equal(seen.le_.classes_, np.array([0, 1]))

    multi_callback = _RootSnapshot()
    multi = MultiOutputClassifier(
        LogisticRegression(max_iter=200), n_jobs=2
    ).set_callbacks(multi_callback)
    with parallel_backend(backend):
        multi.fit(X, y_multi)
    assert len(multi_callback.root_ends) == 1
    classes = multi_callback.root_ends[0].classes_
    assert len(classes) == 2
    assert_array_equal(classes[0], np.array([0, 1]))
    _assert_clean(voting)
    _assert_clean(multi)


def test_partial_fit_predict_and_transform_do_not_emit_callbacks():
    """Only fit and fit_transform participate. Later calls stay silent."""
    X, y = make_regression(n_samples=12, n_features=3, n_targets=2, random_state=0)
    callback = RecordingCallback()
    regressor = MultiOutputRegressor(Ridge(), n_jobs=2).set_callbacks(callback)
    regressor.fit(X, y)
    begin = callback.count_hooks("on_fit_task_begin")
    regressor.predict(X)
    assert callback.count_hooks("on_fit_task_begin") == begin
    assert callback.count_hooks("setup") == 1

    columns = ColumnTransformer(
        [("scale", StandardScaler(), [0, 1])],
        remainder="drop",
    ).set_callbacks(callback)
    columns.fit(X)
    begin = callback.count_hooks("on_fit_task_begin")
    columns.transform(X)
    assert callback.count_hooks("on_fit_task_begin") == begin

    partial = MultiOutputRegressor(SGDRegressor(max_iter=20), n_jobs=2).set_callbacks(
        RecordingCallback()
    )
    partial.partial_fit(X, y)
    assert partial._skl_callbacks[0].count_hooks("setup") == 0
    partial.fit(X, y)
    assert partial._skl_callbacks[0].count_hooks("setup") == 1
    partial.predict(X)
    assert partial._skl_callbacks[0].count_hooks("setup") == 1
