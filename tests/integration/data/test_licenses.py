from pystiche import data


def test_licenses_repr_smoke(subtests):
    licenses = (
        data.UnknownLicense(),
        data.NoLicense(),
        data.PublicDomainLicense(),
        data.ExpiredCopyrightLicense(1970),
        data.PixabayLicense(),
        data.CreativeCommonsLicense(("by", "sa"), "3.0"),
    )
    for license in licenses:
        with subtests.test(license=license):
            assert isinstance(repr(license), str)
