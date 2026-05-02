import requests
from tqdm import tqdm

url = "https://zenodo.org/records/2650142/files/DREAMS%20Subjects%20Database.zip"
output = r"D:\EEG-Sleep-GNN\data\raw\dreams.zip"

headers = {"User-Agent": "Mozilla/5.0"}

response = requests.get(url, stream=True, headers=headers)
total = int(response.headers.get('content-length', 0))

with open(output, 'wb') as file, tqdm(
    desc="Downloading DREAMS",
    total=total,
    unit='iB',
    unit_scale=True
) as bar:
    for data in response.iter_content(chunk_size=1024):
        size = file.write(data)
        bar.update(size)

print("Download complete ✅")