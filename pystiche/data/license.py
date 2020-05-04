from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Sequence

__all__ = [
    "License",
    "UnknownLicense",
    "NoLicense",
    "PublicDomainLicense",
    "ExpiredCopyrightLicense",
    "PixabayLicense",
    "CreativeCommonsLicense",
]


class License(ABC):
    @abstractmethod
    def __repr__(self) -> str:
        pass


class UnknownLicense(License):
    def __repr__(self) -> str:
        return (
            "The license for this image is unknown. "
            "Proceed to work with this image at your own risk."
        )


class NoLicense(License):
    def __repr__(self) -> str:
        return (
            "There is no license available for this image. "
            "Proceed to work with this image at your own risk."
        )


class PublicDomainLicense(License):
    def __repr__(self) -> str:
        return (
            "The copyright holder released this image into the public domain. This "
            "grants anyone the right to use this image for any purpose, without any "
            "conditions, unless such conditions are required by law."
        )


class ExpiredCopyrightLicense(PublicDomainLicense):
    def __init__(self, author_death_year: int):
        self.author_death_year = author_death_year

    def __repr__(self) -> str:
        years_since_author_death = str(datetime.now().year - self.author_death_year)
        return (
            f"This image is in the public domain in countries and areas, "
            f"where the copyright term is at most the authors life plus "
            f"{years_since_author_death} years. Check your local laws before "
            f"proceeding."
        )


class PixabayLicense(License):
    def __repr__(self) -> str:
        return "Pixabay (https://pixabay.com/service/license/)"


class CreativeCommonsLicense(License):
    TYPE_DICT = {
        "by": "Attribution",
        "sa": "ShareAlike",
        "nc": "NonCommercial",
        "nd": "NoDerivatives",
    }

    def __init__(
        self, types: Sequence[str], version: str, variant: Optional[str] = None,
    ) -> None:
        self.types = [type.lower() for type in types]
        self.version = version
        if variant is None:
            if version == "2.0":
                variant = "Generic"
            elif version == "3.0":
                variant = "Unported"
        self.variant = variant

    def __repr__(self) -> str:
        return self._create_license()

    def _create_license(self) -> str:
        long = "{0} {1}".format(
            "-".join([self.TYPE_DICT[type] for type in self.types]), self.version
        )
        if self.variant:
            long += f" {self.variant}"
        short = f"(CC {'-'.join([type.upper() for type in self.types])} {self.version})"
        url = "https://creativecommons.org/licenses/{0}/{1}".format(
            "-".join(self.types), self.version
        )
        return f"{long} {short} {url}"
