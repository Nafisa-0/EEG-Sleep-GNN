import requests
from tqdm import tqdm

url = "https://physionet.org/static/published-projects/sleep-edfx/sleep-edfx-1.0.0.zip"
output = r"D:\EEG-Sleep-GNN\data\raw\sleep-edf.zip"

response = requests.get(url, stream=True)
total = int(response.headers.get('content-length', 0))

with open(output, 'wb') as file, tqdm(
    desc="Downloading",
    total=total,
    unit='iB',
    unit_scale=True
) as bar:
    for data in response.iter_content(chunk_size=1024):
        size = file.write(data)
        bar.update(size)

print("Download complete ✅")