"""Model helper regression tests."""

from app.models import _to_vector


def test_to_vector_flattens_single_row_batch():
    assert _to_vector([[0.1, 0.2, 0.3]]) == [0.1, 0.2, 0.3]


def test_to_vector_accepts_one_dimensional_values():
    assert _to_vector([0.1, 0.2, 0.3]) == [0.1, 0.2, 0.3]
