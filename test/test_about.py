import unittest
import pystiche


class Tester(unittest.TestCase):
    def test_about(self):
        for attr in (
            "name",
            "description",
            "version",
            "url",
            "license",
            "author",
            "author_email",
        ):
            self.assertIsInstance(getattr(pystiche, f"__{attr}__"), str)


if __name__ == "__main__":
    unittest.main()
