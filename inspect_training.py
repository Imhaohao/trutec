"""Inspect only public training queries and reviewed texture examples."""
from pathlib import Path
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw

from evaluate import reviewed_domains


def reviewed_crop(gray: np.ndarray, domain: np.ndarray) -> Image.Image:
    distance = cv2.distanceTransform(domain.astype(np.uint8), cv2.DIST_L2, 3)
    y, x = np.unravel_index(distance.argmax(), distance.shape)
    radius = 24
    patch = gray[max(0, y-radius):y+radius, max(0, x-radius):x+radius]
    return Image.fromarray(patch).resize((144, 144))


def main() -> None:
    root = Path('dataset')
    rows = json.loads((root/'train.json').read_text())['examples']
    rows = [row for row in rows if row['kind'] == 'real'][:24]
    sheet = Image.new('RGB', (720, len(rows)*168), 'white')
    draw = ImageDraw.Draw(sheet)
    summary = []
    for index, row in enumerate(rows):
        gray = cv2.imread(str(root/row['image']), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError('Unreadable training image: '+row['image'])
        positive, known, _, metadata = reviewed_domains(row, root)
        x0, y0, x1, y1 = row['query_box']
        query = Image.fromarray(gray[y0:y1, x0:x1]).resize((144,144))
        sheet.paste(query, (0, index*168+24))
        sheet.paste(reviewed_crop(gray, positive), (160, index*168+24))
        sheet.paste(reviewed_crop(gray, known & ~positive), (320, index*168+24))
        draw.text((0,index*168), row['id']+'   query / positive / negative', fill='black')
        summary.append({'id':row['id'], 'shape':gray.shape, 'query':row['query_box'],
                        'positive':int(positive.sum()), 'negative':int((known & ~positive).sum()),
                        **metadata})
        print(row['id'], flush=True)
    Path('artifacts/inspection').mkdir(parents=True, exist_ok=True)
    sheet.save('artifacts/inspection/training-textures.png')
    Path('artifacts/inspection/training-summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
