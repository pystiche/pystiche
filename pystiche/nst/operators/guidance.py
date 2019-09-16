from ._base import GuidedEncodingComparison
from .encoding import GramEncodingComparisonOperator

__all__ = ["GuidedGramEncodingComparisonOperator"]


class GuidedGramEncodingComparisonOperator(
    GuidedEncodingComparison, GramEncodingComparisonOperator
):
    pass
