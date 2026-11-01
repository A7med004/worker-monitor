"""Download and verify the project's pinned model assets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import urllib.request

from filelock import FileLock, Timeout

PROJECT_ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((PROJECT_ROOT / 'model_manifest.json').read_text(encoding='utf-8'))


def model_root():
    """Use project-local weights unless an explicit cache path is configured."""
    configured = os.environ.get('WORKER_MODELS_ROOT')
    return Path(configured).expanduser().resolve() if configured else PROJECT_ROOT / 'weights'


def valid_asset(path, asset):
    if not path.is_file() or path.stat().st_size != asset['size']:
        return False
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest() == asset['sha256']


def download_asset(asset, root, progress=print, opener=None):
    """Resume interrupted downloads, verify SHA-256, then replace atomically."""
    opener = opener or urllib.request.urlopen
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / asset['name']
    partial = root / (asset['name'] + '.part')
    try:
        with FileLock(str(target) + '.lock', timeout=1):
            if valid_asset(target, asset):
                return target
            if partial.exists() and partial.stat().st_size >= asset['size']:
                partial.unlink()
            for attempt in range(3):
                try:
                    offset = partial.stat().st_size if partial.exists() else 0
                    headers = {'User-Agent': 'Worker-Monitor', 'Accept-Encoding': 'identity'}
                    if offset:
                        headers['Range'] = f'bytes={offset}-'
                    request = urllib.request.Request(asset['url'], headers=headers)
                    progress(f"Downloading {asset['name']} ({asset['size']/1024**2:.1f} MiB)")
                    with opener(request, timeout=30) as response:
                        resumed = offset > 0 and response.status == 206
                        if resumed:
                            expected = f"bytes {offset}-{asset['size']-1}/{asset['size']}"
                            if response.headers.get('Content-Range') != expected:
                                raise ValueError('Server returned an unexpected byte range')
                        elif response.status != 200:
                            raise ValueError(f'Unexpected HTTP status: {response.status}')
                        with partial.open('ab' if resumed else 'wb') as output:
                            last_report = time.monotonic()
                            while block := response.read(1024 * 1024):
                                output.write(block)
                                if output.tell() > asset['size']:
                                    raise ValueError('Downloaded file exceeds expected size')
                                if time.monotonic()-last_report >= 5:
                                    progress(f"{asset['name']}: {output.tell()/asset['size']:.0%}")
                                    last_report = time.monotonic()
                    if not valid_asset(partial, asset):
                        if partial.stat().st_size >= asset['size']:
                            partial.unlink()
                        raise ValueError('Downloaded file is incomplete or failed SHA-256 verification')
                    os.replace(partial, target)
                    progress(f"Verified {asset['name']}")
                    return target
                except Exception as error:
                    if attempt == 2:
                        raise RuntimeError(
                            f"Could not download {asset['name']}: {error}. "
                            'Retry download_models.py to resume, or download the file from MODELS.md.'
                        ) from error
                    time.sleep(1)
    except Timeout as error:
        raise RuntimeError(f"Another process is downloading {asset['name']}. Try again when it finishes.") from error


def ensure_models(root=None, offline=None, progress=print):
    root = Path(root) if root is not None else model_root()
    if offline is None:
        offline = os.environ.get('WORKER_MONITOR_OFFLINE', '').lower() in ('1', 'true', 'yes')
    for asset in MANIFEST['assets']:
        if valid_asset(root / asset['name'], asset):
            continue
        if offline:
            raise RuntimeError(f"Missing or invalid model: {asset['name']}. Run python download_models.py before going offline.")
        download_asset(asset, root, progress)
    return root


def import_models(source, root, progress=print):
    """Import matching weights from an explicitly chosen local directory."""
    source, root = Path(source), Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for asset in MANIFEST['assets']:
        target = root / asset['name']
        if valid_asset(target, asset):
            continue
        for candidate in (source / asset['name'], source / 'models' / asset['name']):
            if valid_asset(candidate, asset):
                with FileLock(str(target) + '.lock', timeout=1):
                    staging = root / (asset['name'] + '.import')
                    shutil.copyfile(candidate, staging)
                    if not valid_asset(staging, asset):
                        staging.unlink()
                        raise ValueError(f"Import failed verification: {asset['name']}")
                    os.replace(staging, target)
                progress(f"Imported and verified {asset['name']}")
                break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dir', type=Path, help='Override the model cache directory')
    parser.add_argument('--check', action='store_true', help='Verify files without downloading')
    parser.add_argument('--import-from', dest='source', type=Path, help='Import matching local weights before downloading missing assets')
    parser.add_argument('--offline', action='store_true', help='Never download missing assets')
    args = parser.parse_args()
    root = args.dir.resolve() if args.dir else model_root()
    if args.check:
        checks = [(a, valid_asset(root/a['name'], a)) for a in MANIFEST['assets']]
        for asset, ok in checks:
            print(f"{'OK' if ok else 'MISSING OR INVALID'}  {asset['name']}")
        return 0 if all(ok for _, ok in checks) else 1
    try:
        if args.source:
            import_models(args.source, root)
        ensure_models(root, offline=True if args.offline else None)
        print(f'Models ready: {root}')
        return 0
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
