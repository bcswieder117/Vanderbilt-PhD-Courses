"""Optional: regenerate the packaged Silverbox CSVs from the original public archive."""
from pathlib import Path
from urllib.request import Request,urlopen
from zipfile import ZipFile
import csv,hashlib,io,json
import numpy as np
from scipy.io import loadmat
ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'data/source_manifest.json').read_text())
print('Downloading the original Silverbox archive...',flush=True)
with urlopen(Request(manifest['download_url'],headers={'User-Agent':'Silverbox classroom reproduction'}),timeout=60)as response:
 archive=response.read()
assert hashlib.sha256(archive).hexdigest()==manifest['source_archive_sha256'],'Source archive differs from the recorded version'
with ZipFile(io.BytesIO(archive))as z:
 values=loadmat(io.BytesIO(z.read('SilverboxFiles/SNLS80mV.mat')))
u=values['V1'].ravel();y=values['V2'].ravel();fs=manifest['sampling_frequency_Hz']
for name,record in manifest['classroom_subsets'].items():
 path=ROOT/'data'/record['file']
 with path.open('w',newline='')as f:
  w=csv.writer(f);w.writerow(['sample_index','time_s','input_voltage_V','output_voltage_V'])
  for i in range(record['raw_start_inclusive'],record['raw_stop_exclusive']):
   w.writerow([i,format(i/fs,'.17g'),format(u[i],'.17g'),format(y[i],'.17g')])
 assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256'],f'CSV differs: {name}'
 print('Recreated and verified:',path.name,record['rows'],'rows')
print('All four CSV files match the recorded source version.')
