import pytest

from possession import registry


def test_duplicate_name_is_rejected():
    with pytest.raises(ValueError, match="already registered"):

        @registry.register("decode", "majority_vote")
        def again(match, p_alive, p_home):
            """Duplicate."""


def test_docstring_is_required():
    with pytest.raises(ValueError, match="docstring"):

        @registry.register("team", "_test_no_doc")
        def no_doc(match):
            pass


def test_parameters_need_defaults():
    with pytest.raises(TypeError, match="defaults"):

        @registry.register("team", "_test_no_default")
        def no_default(match, radius):
            """Missing default."""


def test_unknown_stage():
    with pytest.raises(ValueError, match="unknown stage"):
        registry.register("possession", "x")


def test_every_stage_has_a_method():
    for stage in registry.STAGES:
        assert registry.methods(stage), f"no method registered for {stage}"
