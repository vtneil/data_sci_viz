import json
from pprint import pprint
from typing import TypedDict, List, AnyStr, Dict


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


PaperEntries = Dict[AnyStr, Paper]


class PaperFactory:
    @staticmethod
    def from_json(path: AnyStr) -> Paper:
        with open(path, mode='r') as fro:
            data = json.load(fro)
        return data

    @staticmethod
    def many_from_json(path: AnyStr) -> PaperEntries:
        with open(path, mode='r') as fro:
            data = json.load(fro)
        return data


if __name__ == '__main__':
    p1 = PaperFactory.from_json('../data/format.json')
    p2 = PaperFactory.many_from_json('../data/sample.json')

    pprint(p1)
    pprint(p2)
