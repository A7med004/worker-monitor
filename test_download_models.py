"""Tests for verified model downloads; no external network access required."""
import hashlib
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import download_models as models


class Response(BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {}


class ModelDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = b'verified-model-content'
        self.asset = dict(name='model.pt', size=len(self.data),
                          sha256=hashlib.sha256(self.data).hexdigest(),
                          url='https://example.invalid/model.pt')

    def fetch(self, opener):
        return models.download_asset(self.asset, self.root, progress=lambda _: None, opener=opener)

    def test_download_and_reuse_verified_file(self):
        path = self.fetch(lambda *a, **k: Response(self.data))
        self.assertEqual(path.read_bytes(), self.data)
        self.fetch(lambda *a, **k: self.fail('A verified cache must not download again'))

    def test_resume_partial_download(self):
        (self.root/'model.pt.part').write_bytes(self.data[:5])
        def opener(request, **kwargs):
            self.assertEqual(request.get_header('Range'), 'bytes=5-')
            return Response(self.data[5:], 206,
                            {'Content-Range':f'bytes 5-{len(self.data)-1}/{len(self.data)}'})
        self.assertEqual(self.fetch(opener).read_bytes(), self.data)

    def test_restart_when_server_ignores_range(self):
        (self.root/'model.pt.part').write_bytes(b'old-partial')
        self.assertEqual(self.fetch(lambda *a, **k: Response(self.data)).read_bytes(), self.data)

    def test_checksum_failure_does_not_replace_existing_file(self):
        target=self.root/'model.pt'
        target.write_bytes(b'previous-file')
        with patch.object(models.time, 'sleep'):
            with self.assertRaises(RuntimeError):
                self.fetch(lambda *a, **k: Response(b'x'*len(self.data)))
        self.assertEqual(target.read_bytes(), b'previous-file')

    def test_offline_mode_rejects_missing_assets(self):
        with patch.object(models, 'MANIFEST', {'assets':[self.asset]}):
            with patch.object(models.urllib.request, 'urlopen') as network:
                with self.assertRaisesRegex(RuntimeError, 'Missing or invalid'):
                    models.ensure_models(self.root, offline=True)
                network.assert_not_called()

    def test_import_copies_matching_local_assets(self):
        source=self.root/'source'
        (source/'models').mkdir(parents=True)
        (source/'models/model.pt').write_bytes(self.data)
        cache=self.root/'cache'
        with patch.object(models, 'MANIFEST', {'assets':[self.asset]}):
            models.import_models(source,cache,progress=lambda _:None)
        self.assertTrue(models.valid_asset(cache/'model.pt',self.asset))


if __name__ == '__main__':
    unittest.main()
