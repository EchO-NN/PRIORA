"""Retrieve and verify the official instruction and scoring text for one task."""
import hashlib
import json
from pathlib import Path

REGISTRY = Path(__file__).parent/'robodojo/wall_oss/cards.json'


def load_card(path, task):
    card = json.loads(Path(path).read_text())
    expected = json.loads(REGISTRY.read_text())[task]
    actual = hashlib.sha256(card['text'].encode()).hexdigest()
    if card['task'] != task or actual != expected['sha256'] or actual != card['sha256']:
        raise ValueError(f'Official scoring card differs from the evaluation version for {task}')
    return card


def fetch(task, destination):
    import requests
    from bs4 import BeautifulSoup
    from .robodojo.io import write_json
    url = json.loads(REGISTRY.read_text())[task]['url']
    response = requests.get(url, timeout=(10, 45))
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')

    def section(name):
        header = soup.select_one('main h2#'+name)
        if header is None:
            raise ValueError(f'Missing official section {name}')
        for tag in header.parent.find_next_siblings():
            if tag.name == 'h2' or tag.select_one('h2'):
                break
            yield tag

    detail = [tag.get_text(' ', strip=True) for tag in section('detailed-instruction')
              if tag.name in ('p','ul','ol')]
    scoring = [[' '.join(cell.stripped_strings) for cell in row.select('th,td')]
               for tag in section('scoring') for row in tag.select('tr')]
    text = 'Detailed Instruction\n'+'\n'.join(detail)+'\n\nScoring\n'+'\n'.join(' | '.join(row) for row in scoring)
    digest = hashlib.sha256(text.encode()).hexdigest()
    if digest != json.loads(REGISTRY.read_text())[task]['sha256']:
        raise ValueError('The published card changed. Supply the pinned card file for this evaluation.')
    write_json(destination, dict(task=task, official_url=url, detail=detail, scoring=scoring,
                                 text=text, sha256=digest))
