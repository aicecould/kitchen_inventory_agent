"""Download pinned Qwen weights directly from ModelScope, bypassing proxies."""

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / 'models' / 'Qwen3-1.7B'
CHUNK_SIZE = 16 * 1024 * 1024


def fetch(url, target, start=None, end=None):
    headers = {'User-Agent': 'qwen-local-deployment/1.0', 'Accept-Encoding': 'identity'}
    if start is not None:
        headers['Range'] = f'bytes={start}-{end}'
    # Explicit empty ProxyHandler also ignores lowercase and ALL_PROXY settings.
    for attempt in range(3):
        try:
            offset = target.stat().st_size if start is not None and target.exists() else 0
            actual_start = start + offset if start is not None else None
            if actual_start is not None:
                headers['Range'] = f'bytes={actual_start}-{end}'
            with build_opener(ProxyHandler({})).open(Request(url, headers=headers), timeout=60) as response:
                if start is not None:
                    expected = f'bytes {actual_start}-{end}/'
                    if response.status != 206 or not response.headers.get('Content-Range', '').startswith(expected):
                        raise RuntimeError('Mirror did not honor the requested byte range')
                with target.open('ab' if offset else 'wb') as output:
                    while block := response.read(1024 * 1024):
                        output.write(block)
            if start is not None and target.stat().st_size != end - start + 1:
                raise RuntimeError('Incomplete download range')
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        while block := source.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def download_file(entry, manifest):
    name, size = entry['path'], entry['size']
    target = MODEL_DIR / name
    if target.exists() and target.stat().st_size == size and sha256(target) == entry['sha256']:
        print(f'Verified {name}', flush=True)
        return
    url = f"{manifest['endpoint']}/models/{manifest['repo_id']}/resolve/{manifest['revision']}/{name}"
    temporary = target.with_suffix(target.suffix + '.partial')
    if size > CHUNK_SIZE:
        parts = MODEL_DIR / '.parts' / name
        parts.mkdir(parents=True, exist_ok=True)
        ranges = [(start, min(start + CHUNK_SIZE, size) - 1) for start in range(0, size, CHUNK_SIZE)]

        def part(bounds):
            start, end = bounds
            path = parts / str(start)
            if not path.exists() or path.stat().st_size != end - start + 1:
                fetch(url, path, start, end)
            print(f'{name}: {end + 1}/{size}', flush=True)
            return path

        with ThreadPoolExecutor(max_workers=32) as pool:
            paths = list(pool.map(part, ranges))
        with temporary.open('wb') as output:
            for path in paths:
                with path.open('rb') as source:
                    while block := source.read(4 * 1024 * 1024):
                        output.write(block)
    else:
        fetch(url, temporary)
    if temporary.stat().st_size != size or sha256(temporary) != entry['sha256']:
        raise RuntimeError(f'Size or SHA-256 mismatch: {name}')
    temporary.replace(target)
    if size > CHUNK_SIZE:
        for path in paths:
            path.unlink()
        parts.rmdir()
    print(f'Verified {name}', flush=True)


if __name__ == '__main__':
    manifest = json.loads((ROOT / 'model-manifest.json').read_text())
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for entry in manifest['files']:
        download_file(entry, manifest)
    (MODEL_DIR / 'source.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'All files verified: {MODEL_DIR}', flush=True)
