"""rag.store.point_id: deterministic point IDs are what make re-ingestion
after a partial failure safe (overwrite, not duplicate)."""

from rag.store import point_id


def test_same_input_yields_same_id():
    assert point_id("techniques/T1539.txt", 0) == point_id("techniques/T1539.txt", 0)


def test_different_chunk_index_yields_different_id():
    assert point_id("techniques/T1539.txt", 0) != point_id("techniques/T1539.txt", 1)


def test_different_source_yields_different_id():
    assert point_id("a.txt", 0) != point_id("b.txt", 0)


def test_id_is_a_valid_uuid_string():
    import uuid

    uuid.UUID(point_id("a.txt", 0))  # raises ValueError if malformed
