import unittest

from audit_agent.redact import find_secrets, redact_text
from audit_agent.repo import InvalidRepositoryURL, parse_github_url
from tests.helpers import FAKE_AWS_KEY


class RedactionTests(unittest.TestCase):
    def test_known_key_formats_are_redacted(self):
        out = redact_text(f'AWS_KEY = "{FAKE_AWS_KEY}"')
        self.assertNotIn(FAKE_AWS_KEY, out)
        self.assertIn("[REDACTED:aws_access_key_id]", out)

    def test_hardcoded_password_redacted(self):
        secret = "hunter2" + "supersecret"
        out = redact_text(f'DB_PASSWORD = "{secret}"')
        self.assertNotIn(secret, out)

    def test_connection_string_password_redacted(self):
        out = redact_text("postgres://admin:" + "S3cr3tPass" + "@db.internal:5432/app")
        self.assertNotIn("S3cr3tPass", out)
        self.assertIn("admin:", out)

    def test_private_key_header_redacted(self):
        text = "-----BEGIN RSA PRIVATE KEY-----\nMIIEow..."
        self.assertIn("[REDACTED]", redact_text(text))

    def test_placeholders_are_not_secrets(self):
        for line in ('PASSWORD = "changeme"', "API_KEY=${API_KEY}", "TOKEN=your-token-here",
                     'password = "{password}"'):
            hits = find_secrets(line)
            self.assertTrue(all(h.placeholder for h in hits), line)
            self.assertEqual(redact_text(line), line)

    def test_plain_text_untouched(self):
        text = "def add(a, b):\n    return a + b\n"
        self.assertEqual(redact_text(text), text)


class UrlTests(unittest.TestCase):
    def test_valid_urls(self):
        for url in ("https://github.com/psf/requests", "https://github.com/psf/requests.git",
                    "github.com/psf/requests/", "http://www.github.com/psf/requests"):
            ref = parse_github_url(url)
            self.assertEqual(ref.url, "https://github.com/psf/requests")
            self.assertEqual((ref.owner, ref.name), ("psf", "requests"))

    def test_invalid_urls(self):
        for url in ("", "not a url", "https://gitlab.com/a/b", "https://github.com/onlyowner",
                    "file:///etc/passwd", "https://github.com/a/b/tree/main", "--upload-pack=x",
                    "https://github.com/a/.."):
            with self.assertRaises(InvalidRepositoryURL, msg=url):
                parse_github_url(url)


if __name__ == "__main__":
    unittest.main()
