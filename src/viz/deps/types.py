import time
from typing import TypedDict, List, AnyStr

import ujson as json

from src.viz.deps.utils import benchmark


class Location(TypedDict):
    lat: AnyStr
    lng: AnyStr


class Affiliation(TypedDict):
    name: AnyStr
    location: Location
    city: AnyStr
    country: AnyStr


class Paper(TypedDict):
    SCOPUSID: AnyStr
    title: AnyStr
    affiliations: List[Affiliation]
    abbrevs: List[AnyStr]
    authors: List[AnyStr]
    publish_year: AnyStr


PaperEntries = List[Paper]


class PaperFactory:
    @staticmethod
    def from_json(path: AnyStr) -> Paper:
        with open(path, mode='r') as fro:
            data = json.load(fro)
        return data

    @staticmethod
    @benchmark
    def many_from_json(path: AnyStr, count: int = -1) -> PaperEntries:
        with open(path, mode='r', encoding='utf-8') as fro:
            raw = json.load(fro)
        if count < 0:
            return raw['all_paper_with_loc']
        else:
            return raw['all_paper_with_loc'][:count]


if __name__ == '__main__':
    # p1 = PaperFactory.from_json('../data/format.json')
    # pprint(p1)
    t = time.time()
    p2 = PaperFactory.many_from_json('../data/papers.json', 10)
    print(time.time() - t)

    print(len(p2))
    s = sum(len(p['affiliations']) * (len(p['affiliations']) - 1) for p in p2)
    print(s)
    # pprint(p2)
