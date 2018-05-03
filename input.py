import urllib.request
import os.path
import tarfile


def _check_if_exists(name):
    return os.path.exists(name)


def _download(url):
    filename = url.split('/')[-1]
    if not _check_if_exists(filename):
        urllib.request.urlretrieve(url, filename)
    return filename


def _extract(filename):
    tarfile.open(filename).extractall()


def extract_if_needed(name, url):
    if _check_if_exists(name):
        return

    filename = _download(url)
    _extract(filename)
