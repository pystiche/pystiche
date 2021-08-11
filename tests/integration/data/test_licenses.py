from pystiche import data


class TestLicenses:
    def test_repr_smoke(self, subtests):
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
