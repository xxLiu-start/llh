"""Serve a single explicitly prepared external fixture through urllib transport.

Test inputs are the original downloaded bytes, hash recorded in the manifest.
All other URLs (including task localhost services) keep urllib's behavior.
"""
import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path
from email.message import Message


def install(root):
    path = Path(root) / "system/manifest.json"
    if not path.exists():
        return
    assets = json.loads(path.read_text())["assets"]
    mapping = {record["url"]: Path(root) / "system/assets" / name / "payload"
               for name, record in assets.items() if name == "filter-js-from-html"}
    original = urllib.request.urlretrieve

    def retrieve(url, filename=None, reporthook=None, data=None):
        if url not in mapping or data is not None:
            return original(url, filename, reporthook, data)
        if filename is None:
            fd, filename = tempfile.mkstemp()
            os.close(fd)
        try:
            shutil.copyfile(mapping[url], filename)
        except OSError:
            from dispatch import fail
            fail("prepared external verifier fixture is unreadable; no online fallback")
        headers = Message()
        headers["Content-Length"] = str(mapping[url].stat().st_size)
        if reporthook:
            reporthook(1, mapping[url].stat().st_size, mapping[url].stat().st_size)
        return str(filename), headers

    urllib.request.urlretrieve = retrieve
