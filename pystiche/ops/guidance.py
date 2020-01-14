from ._base import GuidedEncodingComparisonOperator
from .encoding import GramEncodingComparisonOperator

__all__ = ["GuidedGramEncodingComparisonOperator"]


class GuidedGramEncodingComparisonOperator(
    GuidedEncodingComparisonOperator, GramEncodingComparisonOperator
):
    pass
