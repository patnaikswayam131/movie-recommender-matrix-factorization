"""Fetch the original GroupLens archive. Raw data is not redistributed."""
from pathlib import Path
import hashlib
import io
import urllib.request
import zipfile
import numpy as np

URL = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
ROOT = Path(__file__).resolve().parents[1]


def fetch_data():
    target = ROOT / 'data' / 'ml-100k'
    if not (target / 'u.data').exists():
        print('Downloading MovieLens 100K from GroupLens. Research/education use only.')
        request = urllib.request.Request(URL, headers={'User-Agent': 'MDS-Educational-Recommender/1.0'})
        with urllib.request.urlopen(request, timeout=90) as response:
            raw = response.read()
        target.mkdir(parents=True, exist_ok=True)
        # Read named members, never extract arbitrary archive paths.
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            for name in ('u.data', 'u.item', 'README'):
                (target / name).write_bytes(archive.read('ml-100k/' + name))
        (ROOT / 'data' / 'archive_sha256.txt').write_text(hashlib.sha256(raw).hexdigest() + '\n')
    ratings = np.loadtxt(target / 'u.data', dtype=np.int64)
    if ratings.shape != (100000, 4) or not np.all((ratings[:, 2] >= 1) & (ratings[:, 2] <= 5)):
        raise ValueError('Unexpected MovieLens 100K file contents')
    if len(np.unique(ratings[:, :2], axis=0)) != len(ratings):
        raise ValueError('Duplicate user-item ratings')
    ratings[:, :2] -= 1  # Internal IDs zero-based; UI IDs one-based.
    titles = {}
    for line in (target / 'u.item').read_text(encoding='latin-1').splitlines():
        columns = line.split('|')
        titles[int(columns[0]) - 1] = columns[1]
    return ratings, titles
