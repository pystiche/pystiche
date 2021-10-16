import pytest

from pystiche import data


@pytest.mark.parametrize(
    "license",
    (
        data.UnknownLicense(),
        data.NoLicense(),
        data.PublicDomainLicense(),
        data.ExpiredCopyrightLicense(1970),
        data.PixabayLicense(),
        data.CreativeCommonsLicense(("by", "sa"), "3.0"),
    ),
    ids=lambda license: type(license).__name__,
)
def test_licenses_repr_smoke(license):
    assert isinstance(repr(license), str)
