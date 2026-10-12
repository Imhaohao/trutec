"""Support gates reject insufficient reviewed dual-class query coverage."""
import pytest

from tools.cadence_fit_support import support_gate


@pytest.mark.parametrize("queries,documents,passed", [(29, 10, False), (30, 9, False), (30, 10, True)])
def test_support_requires_both_query_and_document_minima(queries, documents, passed):
    records = [
        {"stable": True, "dual_class": True, "document_id": str(i % documents)}
        for i in range(queries)
    ]
    assert support_gate(records)["passed"] is passed


def test_unavailable_and_positive_only_queries_cannot_rescue_support():
    records = [{"stable": True, "dual_class": True, "document_id": str(i % 10)} for i in range(29)]
    records += [{"stable": False, "dual_class": True, "document_id": "unavailable"}]
    records += [{"stable": True, "dual_class": False, "document_id": "positive-only"}]
    result = support_gate(records)
    assert result["stable_queries"] == 30
    assert result["stable_dual_queries"] == 29
    assert result["stable_dual_documents"] == 10
    assert result["passed"] is False
