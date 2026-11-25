"""Extract verified Ubuntu Python headers locally, without sudo/system changes."""
import hashlib
import subprocess
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parents[2]
target=ROOT/'.local-tools/python3.10-headers'
target.mkdir(parents=True,exist_ok=True)
metadata=subprocess.check_output(['apt-cache','show','libpython3.10-dev=3.10.12-1~22.04.18'],text=True)
fields=dict(line.split(': ',1) for line in metadata.splitlines() if ': ' in line and not line.startswith(' '))
url='https://security.ubuntu.com/ubuntu/'+fields['Filename']
archive=target/Path(fields['Filename']).name
with requests.get(url,stream=True,timeout=90) as response:
    response.raise_for_status()
    with archive.open('wb') as stream:
        for block in response.iter_content(1024**2):stream.write(block)
if hashlib.sha256(archive.read_bytes()).hexdigest()!=fields['SHA256']:
    raise RuntimeError('Ubuntu package SHA256 mismatch')
subprocess.run(['dpkg-deb','-x',str(archive),str(target)],check=True)
assert (target/'usr/include/python3.10/Python.h').is_file()
print('Verified Ubuntu headers extracted locally:',target)
