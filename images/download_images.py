from os import path
from collections import OrderedDict
from datetime import datetime
import requests
from torchvision.datasets.utils import check_integrity


class LicensedImage:
    # TODO: make this dynamic
    _DESCRIPTION_WIDTH = 10

    def __init__(
        self,
        title,
        author,
        date,
        url,
        license,
        tutorials=(),
        papers=(),
        filename=None,
        md5=None,
    ):
        self.title = title
        self.author = author
        self.date = date
        self.url = url
        self.license = license

        if not tutorials and not papers:
            raise RuntimeError
        self.tutorials = tutorials
        self.papers = papers

        if filename is None:
            filename = "{0}__{1}".format(
                author.split(" ")[-1].lower(),
                "_".join([part.lower() for part in title.split(" ")]),
            )
            fileext = path.splitext(url)[1]
        else:
            fileext = path.splitext(url)[1]
        self.file = filename + fileext
        self.md5 = md5

    def check_integrity(self, root):
        return check_integrity(path.join(root, self.file), self.md5)

    def download(self, root):
        if self.check_integrity(root):
            return

        with open(path.join(root, self.file), "wb") as fh:
            fh.write(requests.get(self.url).content)

    def __str__(self):
        info = OrderedDict(
            [
                ("Title", self.title),
                ("Author", self.author),
                ("Date", self.date),
                ("URL", self.url),
                ("License", self.license),
            ]
        )
        if self.tutorials:
            info["Tutorials"] = ", ".join(self.tutorials)
        if self.papers:
            info["Papers"] = ", ".join(self.papers)
        info["File"] = self.file

        return "\n".join(
            [self._str_line(description, value) for description, value in info.items()]
        )

    @property
    def _description_fmtstr(self):
        return "{:" + str(self._DESCRIPTION_WIDTH) + "s}"

    def _str_line(self, description, value):
        value_lines = [line.strip() for line in value.splitlines()]
        value_line = "{}:  {}".format(
            self._description_fmtstr.format(description), value_lines[0]
        )
        if len(value_lines) == 1:
            return value_line

        indent_width = self._DESCRIPTION_WIDTH + 3
        lines = [value_line] + [
            " " * indent_width + value_line for value_line in value_lines[1:]
        ]
        return "\n".join(lines)


class UnkonwnLicense(LicensedImage):
    def __init__(self, title, author, date, url, **kwargs):
        license = "The license for this image is unknown. Proceed at your own risk."
        super().__init__(title, author, date, url, license, **kwargs)


class NPRGeneralLicense(UnkonwnLicense):
    def __init__(self, title, author, url, **kwargs):
        license = """Although the authors of the dataset, David Mould and Paul Rosin, 
        claim that this license 'permits distribution of modified versions', 
        the actual license is unknown. Proceed at your own risk."""
        date = "unknown"
        super().__init__(title, author, date, url, **kwargs)
        self.license = license


class PublicDomainPainting(LicensedImage):
    def __init__(self, title, author, date, author_death, url, ca=False, **kwargs):
        in_public_domain_for = str(datetime.now().year - author_death)
        license = f"""This is a faithful photographic reproduction of a two-dimensional, 
        public domain work of art. This work is in the public domain in its country of 
        origin and other countries and areas where the copyright term is the author's 
        life plus {in_public_domain_for} years."""
        self.in_public_domain_for = in_public_domain_for
        super().__init__(title, author, date, url, license, **kwargs)


class CreativeCommonsImage(LicensedImage):
    TYPE_DICT = {
        "by": "Attribution",
        "sa": "ShareAlike",
        "nc": "NonCommercial",
        "nd": "NoDerivatives",
    }

    def __init__(
        self, title, author, date, url, types, version, variant=None, **kwargs
    ):
        self.types = [type.lower() for type in types]
        self.version = version
        if variant is None:
            if version == "2.0":
                variant = "Generic"
            elif version == "3.0":
                variant = "Unported"
        self.variant = variant

        license = self._create_license()
        super().__init__(title, author, date, url, license, **kwargs)

    def _create_license(self):
        long = "{0} {1}".format(
            "-".join([self.TYPE_DICT[type] for type in self.types]), self.version
        )
        if self.variant is not None:
            long += " {0}".format(self.variant)
        short = "(CC {0} {1})".format(
            "-".join([type.upper() for type in self.types]), self.version
        )
        url = "https://creativecommons.org/licenses/{0}/{1}".format(
            "-".join(self.types), self.version
        )
        return "{0} {1}\n{2}".format(long, short, url)


images = []

npr_general = [
    NPRGeneralLicense(
        title="angel",
        author="Eole Wind",
        url="http://gigl.scs.carleton.ca/sites/default/files/angel1024.jpg",
        papers=("mould_rosin_2016", "jing_et_al_2018"),
        md5="41b058edd2091eec467d37054a8c01fb",
    ),
    NPRGeneralLicense(
        title="arch",
        author="James Marvin Phelps",
        url="http://gigl.scs.carleton.ca/sites/default/files/arch1024.jpg",
        papers=("mould_rosin_2016",),
        md5="9378e3a129f021548ba26b99ae445ec9",
    ),
    NPRGeneralLicense(
        title="athletes",
        author="Nathan Congleton",
        url="http://gigl.scs.carleton.ca/sites/default/files/athletes1024.jpg",
        papers=("mould_rosin_2016",),
        md5="6b742bd1b46bf9882cc176e0d255adab",
    ),
    NPRGeneralLicense(
        title="barn",
        author="MrClean1982",
        url="http://gigl.scs.carleton.ca/sites/default/files/barn1024.jpg",
        papers=("mould_rosin_2016", "jing_et_al_2018"),
        md5="32abf24dd439940cf9a1265965b5e910",
    ),
    NPRGeneralLicense(
        title="berries",
        author="HelmutZen",
        url="http://gigl.scs.carleton.ca/sites/default/files/berries1024.jpg",
        papers=("mould_rosin_2016",),
        md5="58fea65145fa0bd222800d4f8946d815",
    ),
    NPRGeneralLicense(
        title="cabbage",
        author="Leonard Chien",
        url="http://gigl.scs.carleton.ca/sites/default/files/cabbage1024.jpg",
        papers=("mould_rosin_2016",),
        md5="19304a80d2ca2389153562700f0aab53",
    ),
    NPRGeneralLicense(
        title="cat",
        author="Theen Moy",
        url="http://gigl.scs.carleton.ca/sites/default/files/cat1024.jpg",
        papers=("mould_rosin_2016",),
        md5="de3331050c660e688463fa86c76976c4",
    ),
    NPRGeneralLicense(
        title="city",
        author="Rob Schneider",
        url="http://gigl.scs.carleton.ca/sites/default/files/city1024.jpg",
        papers=("mould_rosin_2016",),
        md5="84e17b078ede986ea3e8f70e0a24195e",
    ),
    NPRGeneralLicense(
        title="daisy",
        author="mgaloseau",
        url="http://gigl.scs.carleton.ca/sites/default/files/daisy1024.jpg",
        papers=("mould_rosin_2016",),
        md5="f3901d987490613238ef01977f9fac77",
    ),
    NPRGeneralLicense(
        title="dark woods",
        author="JB Banks",
        url="http://gigl.scs.carleton.ca/sites/default/files/darkwoods1024.jpg",
        papers=("mould_rosin_2016",),
        md5="f4ccbf37b3d5d3a1ca734bb65464151b",
    ),
    NPRGeneralLicense(
        title="desert",
        author="Charles Roffey",
        url="http://gigl.scs.carleton.ca/sites/default/files/desert1024.jpg",
        papers=("mould_rosin_2016", "jing_et_al_2018"),
        md5="4a9db691d203dd693b14090b9e49f791",
    ),
    NPRGeneralLicense(
        title="headlight",
        author="Photos By Clark",
        url="http://gigl.scs.carleton.ca/sites/default/files/headlight1024.jpg",
        papers=("mould_rosin_2016",),
        md5="1d4723535ea7dee84969f0c082445ad5",
    ),
    NPRGeneralLicense(
        title="Mac",
        author="Martin Kenney",
        url="http://gigl.scs.carleton.ca/sites/default/files/mac1024.jpg",
        papers=("mould_rosin_2016",),
        md5="4df4e0dafe5468b86138839f68beb84d",
    ),
    NPRGeneralLicense(
        title="mountains",
        author="Jenny Pansing",
        url="http://gigl.scs.carleton.ca/sites/default/files/mountains1024.jpg",
        papers=("mould_rosin_2016", "jing_et_al_2018"),
        md5="1ea43310f2d85b271827634c733b6257",
    ),
    NPRGeneralLicense(
        title="Oparara",
        author="trevorklatko",
        url="http://gigl.scs.carleton.ca/sites/default/files/oparara1024.jpg",
        papers=("mould_rosin_2016",),
        md5="1adcc54d9a390b2492ee6e71acac13fd",
    ),
    NPRGeneralLicense(
        title="rim lighting",
        author="Paul Stevenson",
        url="http://gigl.scs.carleton.ca/sites/default/files/rim1024.jpg",
        papers=("mould_rosin_2016",),
        md5="8878a60ccd81e7b2ff05089cdf54773b",
    ),
    NPRGeneralLicense(
        title="snow",
        author="John Anes",
        url="http://gigl.scs.carleton.ca/sites/default/files/snow1024.jpg",
        papers=("mould_rosin_2016", "jing_et_al_2018"),
        md5="60d55728e28d39114d0035c04c9e4495",
    ),
    NPRGeneralLicense(
        title="tomatoes",
        author="Greg Myers",
        url="http://gigl.scs.carleton.ca/sites/default/files/tomato1024.jpg",
        papers=("mould_rosin_2016", "jing_et_al_2018"),
        md5="3ffcdc427060171aabf433181cef7c52",
    ),
    NPRGeneralLicense(
        title="toque",
        author="sicknotepix",
        url="http://gigl.scs.carleton.ca/sites/default/files/toque1024.jpg",
        papers=("mould_rosin_2016",),
        md5="34fcf032b5d87877a7fc95629efb94e8",
    ),
    NPRGeneralLicense(
        title="Yemeni",
        author="Richard Messenger",
        url="http://gigl.scs.carleton.ca/sites/default/files/yemeni1024.jpg",
        papers=("mould_rosin_2016", "meier_lohweg_2019"),
        md5="dd1e81d885cdbdd44f8d504e3951fb48",
    ),
]
images.extend(npr_general)

nst_review_style_images = [
    PublicDomainPainting(
        title="Three Fishing Boats",
        author="Claude Monet",
        date="1885",
        author_death=1926,
        url="https://uploads2.wikiart.org/images/claude-monet/three-fishing-boats.jpg",
        papers=("jing_et_al_2018",),
        md5="f2460137cc8c7945a3dcaa4416812bdf",
    ),
    # # FIXME: find a download link
    # PublicDomainPainting(
    #     title="Head of a Clown",
    #     author="Georges Rouault",
    #     date="ca. 1907",
    #     author_death=1958,
    #     url="",
    #     papers=("jing_et_al_2018",),
    #     md5="",
    # ),
    PublicDomainPainting(
        title="Divan Japonais ",
        author="Henri de Toulouse-Lautrec",
        date="ca. 1893",
        author_death=1901,
        url="https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/Henri_de_Toulouse-Lautrec_-_Divan_Japonais_-_Google_Art_Project.jpg/1280px-Henri_de_Toulouse-Lautrec_-_Divan_Japonais_-_Google_Art_Project.jpg",
        papers=("jing_et_al_2018",),
        md5="36b1caf33bc40f8acdaa9f897fb6eea8",
    ),
    PublicDomainPainting(
        title="White Zig Zags ",
        author="Wassily Kandinsky",
        date="1922",
        author_death=1944,
        url="http://www.wassily-kandinsky.org/images/gallery/White-Zig-Zags.jpg",
        papers=("jing_et_al_2018",),
        md5="00ebea08adf077b46c3a3d8802845d13",
    ),
    PublicDomainPainting(
        title="Trees in a Lane",
        author="John Ruskin",
        date="1847",
        author_death=1900,
        url="http://www.victorianweb.org/painting/ruskin/drawings/35.jpg",
        papers=("jing_et_al_2018",),
        md5="089b94ce2a04070e3e2b0fd696d4e5b6",
    ),
    PublicDomainPainting(
        title="Ritmo plastico del 14 luglio",
        author="Severini Gino",
        date="1913",
        author_death=1966,
        url="http://www.mart.tn.it/UploadImgs/4440_115124Severini_Gino_Ritmo_plastico_del_14_luglio.jpg",
        papers=("jing_et_al_2018",),
        md5="4b747ae8cc76fe12f41ba45d36f97173",
    ),
    PublicDomainPainting(
        title="Portrait of Pablo Picasso",
        author="Juan Gris",
        date="1912",
        author_death=1927,
        url="https://upload.wikimedia.org/wikipedia/commons/1/18/Juan_Gris_-_Portrait_of_Pablo_Picasso_-_Google_Art_Project.jpg",
        papers=("jing_et_al_2018",),
        md5="4fd3dbe3e98b7c34910b873b8600bd94",
    ),
    PublicDomainPainting(
        title="Landscape at Saint-Remy",
        author="Vincent van Gogh",
        date="1889",
        author_death=1890,
        url="https://upload.wikimedia.org/wikipedia/commons/thumb/8/86/Gogh%2C_Vincent_van_-_Landscape_at_Saint-R%C3%A9my_%28Enclosed_Field_with_Peasant%29_-_Google_Art_Project.jpg/1276px-Gogh%2C_Vincent_van_-_Landscape_at_Saint-R%C3%A9my_%28Enclosed_Field_with_Peasant%29_-_Google_Art_Project.jpg",
        filename="van_gogh__landscape_at_saint-remy",
        papers=("jing_et_al_2018",),
        md5="6bf9e3b98a5f82aa23bd11fa669ac594",
    ),
    PublicDomainPainting(
        title="The Tower of Babel",
        author="Pieter Bruegel the Elder",
        date="1563",
        author_death=1569,
        url="https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Pieter_Bruegel_the_Elder_-_The_Tower_of_Babel_%28Vienna%29_-_Google_Art_Project.jpg/1280px-Pieter_Bruegel_the_Elder_-_The_Tower_of_Babel_%28Vienna%29_-_Google_Art_Project.jpg",
        filename="bruegel__the_tower_of_babel",
        papers=("jing_et_al_2018",),
        md5="33e9258db038ac06b7403fb79691c34a",
    ),
    PublicDomainPainting(
        title="Edith with Striped Dress",
        author="Egon Schiele",
        date="1915",
        author_death=1918,
        url="https://upload.wikimedia.org/wikipedia/commons/c/cd/Egon_Schiele_-_Edith_with_Striped_Dress%2C_Sitting_-_Google_Art_Project.jpg",
        papers=("jing_et_al_2018",),
        md5="dc048bc18ff44971ab290ee83cdefd29",
    ),
]
images.extend(nst_review_style_images)


images.extend(
    [
        CreativeCommonsImage(
            title="Tübingen Neckarfront",
            author="Andreas Praefcke",
            date="February 2003",
            url="https://upload.wikimedia.org/wikipedia/commons/0/00/Tuebingen_Neckarfront.jpg",
            types=("by",),
            version="3.0",
            papers=("gatys_ecker_bethge_2015", "gatys_et_al_2017"),
            filename="praefcke__tuebingen_neckarfront",
            md5="dc9ad203263f34352e18bc29b03e1066",
        ),
        PublicDomainPainting(
            title="Shipwreck of the Minotaur",
            author="J. M. W. Turner",
            date="ca. 1810",
            author_death=1851,
            url="https://upload.wikimedia.org/wikipedia/commons/2/2e/Shipwreck_turner.jpg",
            papers=("gatys_ecker_bethge_2015",),
            md5="b7a51f010b591a01a6769161d1219739",
        ),
        PublicDomainPainting(
            title="Starry Night",
            author="Vincent van Gogh",
            date="ca. 1889",
            author_death=1890,
            url="https://upload.wikimedia.org/wikipedia/commons/d/de/Vincent_van_Gogh_Starry_Night.jpg",
            papers=("gatys_ecker_bethge_2015", "gatys_et_al_2017"),
            filename="van_gogh__starry_night",
            md5="91805d5d6298e329b08451992e63133e",
        ),
        PublicDomainPainting(
            title="The Scream",
            author="Edvard Munch",
            date="ca. 1893",
            author_death=1944,
            url="https://upload.wikimedia.org/wikipedia/commons/f/f4/The_Scream.jpg",
            papers=("gatys_ecker_bethge_2015",),
            md5="46ef64eea5a7b2d13dbadd420b531249",
        ),
        PublicDomainPainting(
            title="Figure dans un Fauteuil",
            author="Pablo Ruiz Picasso",
            date="ca. 1909",
            author_death=1973,
            url="https://upload.wikimedia.org/wikipedia/en/8/8f/Pablo_Picasso%2C_1909-10%2C_Figure_dans_un_Fauteuil_%28Seated_Nude%2C_Femme_nue_assise%29%2C_oil_on_canvas%2C_92.1_x_73_cm%2C_Tate_Modern%2C_London.jpg",
            papers=("gatys_ecker_bethge_2015",),
            md5="ba14b947b225d9e5c59520a814376944",
        ),
        PublicDomainPainting(
            title="Composition VII",
            author="Wassily Kandinsky",
            date="1913",
            author_death=1944,
            url="https://upload.wikimedia.org/wikipedia/commons/b/b4/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg",
            papers=("gatys_ecker_bethge_2015",),
            md5="bfcbc420684bf27d2d8581fa8cc9522f",
        ),
        # FIXME: needs transformation to match image used in the paper
        UnkonwnLicense(
            title="Jeffrey Dennard",
            author="Jeffrey Dennard",
            date="2015",
            url="https://atlantapolyweekend.com/apwwp2017/wp-content/uploads/2015/09/52.jpg",
            papers=("li_wand_2016",),
            filename="jeffrey_dennard",
            md5="cb78e9919d1bbf3569324f172ffb0e73",
        ),
        PublicDomainPainting(
            title="Self-Portrait",
            author="Pablo Ruiz Picasso",
            date="1907",
            author_death=1973,
            url="https://www.pablo-ruiz-picasso.net/images/works/57.jpg",
            papers=("li_wand_2016",),
            filename="picasso__self-portrait_1907",
            md5="082ed6183d2f545a0b7c6e8021588feb",
        ),
        # FIXME: needs transformation to match image used in the paper
        CreativeCommonsImage(
            title="S",
            author="theilr",
            date="2011",
            url="https://live.staticflickr.com/7409/9270411440_cdc2ee9c35_o_d.jpg",
            types=("by", "sa"),
            version="2.0",
            papers=("li_wand_2016",),
            md5="525550983f7fd36d3ec10fba735ad1ef",
        ),
        PublicDomainPainting(
            title="Composition VIII",
            author="Wassily Kandinsky",
            date="1923",
            author_death=1944,
            url="https://www.wassilykandinsky.net/images/works/50.jpg",
            papers=("li_wand_2016",),
            md5="c39077aaa181fd40d7f2cd00c9c09619",
        ),
        UnkonwnLicense(
            title="House Concept Tillamook",
            author="Unknown",
            date="2014",
            url="https://associateddesigns.com/sites/default/files/plan_images/main/craftsman_house_plan_tillamook_30-519-picart.jpg",
            papers=("gatys_et_al_2017",),
            filename="house_concept_tillamook",
            md5="5629bf7b24a7c98db2580ec2a8d784e9",
        ),
        UnkonwnLicense(
            title="Watertown",
            author="Shop602835 Store",
            date="Unknown",
            url="https://ae01.alicdn.com/img/pb/136/085/095/1095085136_084.jpg",
            papers=("gatys_et_al_2017",),
            filename="watertown",
            md5="4cc98a503da5ce6eab0649b09fd3cf77",
        ),
        PublicDomainPainting(
            title="Wheat Field with Cypresses",
            author="Vincent van Gogh",
            date="1889",
            author_death=1890,
            url="https://upload.wikimedia.org/wikipedia/commons/thumb/c/ce/Wheat-Field-with-Cypresses-%281889%29-Vincent-van-Gogh-Met.jpg/1920px-Wheat-Field-with-Cypresses-%281889%29-Vincent-van-Gogh-Met.jpg",
            papers=("gatys_et_al_2017",),
            filename="van_gogh__wheat_field_with_cypresses",
            md5="bfd085d7e800459c8ffb44fa404f73c3",
        ),
        CreativeCommonsImage(
            title="Schultenhof Mettingen Bauerngarten 8",
            author="J.-H. Janßen",
            date="July 2010",
            url="https://upload.wikimedia.org/wikipedia/commons/8/82/Schultenhof_Mettingen_Bauerngarten_8.jpg",
            types=("by", "sa"),
            version="3.0",
            papers=("gatys_et_al_2017",),
            filename="janssen__schultenhof_mettingen_bauerngarten_8",
            md5="23f75f148b7b94d932e599bf0c5e4c8e",
        ),
        PublicDomainPainting(
            title="Starry Night Over the Rhone",
            author="Vincent Willem van Gogh",
            date="1888",
            author_death=1890,
            url="https://upload.wikimedia.org/wikipedia/commons/9/94/Starry_Night_Over_the_Rhone.jpg",
            filename="van_gogh__starry_night_over_rhone",
            papers=("gatys_et_al_2017",),
            md5="406681ec165fa55c26cb6f988907fe11",
        ),
    ]
)

if __name__ == "__main__":
    # TODO: implement switch for download images with unknown licenses
    # TODO: implement switch to download images that have been in the public domain for XXX years
    root = path.join(path.dirname(__file__), "source")

    sep_line = "-" * 80
    for image in images:
        print(image)
        image.download(root)
        print(sep_line)
