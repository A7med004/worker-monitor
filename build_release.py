"""Build a portable source archive with an explicit public-file allowlist."""
import argparse
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
PUBLIC_FILES = (
    '.gitignore', 'README.md', 'MODELS.md', 'THIRD_PARTY.md',
    'requirements.txt', 'requirements-gpu.txt', 'model_manifest.json',
    'app.py', 'monitor.py', 'detectors.py', 'state.py', 'geometry.py',
    'evidence.py', 'annotation.py', 'tracker.yaml', 'download_models.py',
    'build_release.py', 'test_state.py', 'test_evidence.py',
    'test_download_models.py',
)


def build(destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for name in PUBLIC_FILES:
        if not (ROOT/name).is_file():
            raise FileNotFoundError(f'Missing release file: {name}')
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in PUBLIC_FILES:
            archive.write(ROOT/name, 'worker-monitor/'+name)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'dist/worker-monitor.zip')
    args=parser.parse_args()
    print(build(args.out))
