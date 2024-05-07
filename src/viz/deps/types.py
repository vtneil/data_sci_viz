import time
from pprint import pprint
from typing import TypedDict, List, AnyStr

import pandas as pd
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

    @staticmethod
    def entries_to_df(entries: PaperEntries) -> pd.DataFrame:
        cv = pd.json_normalize(
            entries,
            record_path=['affiliations'],
            meta=['SCOPUSID', 'title', 'publish_year', 'abbrevs', 'authors']
        )[['SCOPUSID', 'title', 'publish_year', 'abbrevs', 'authors',
           'name', 'city', 'country', 'location.lat', 'location.lng']].rename(
            columns={
                'SCOPUSID': 'id',
                'publish_year': 'year',
                'location.lat': 'lat',
                'location.lng': 'lng'
            }
        ).dropna()

        numeric_cols = ['year', 'lat', 'lng']

        for col in numeric_cols:
            cv[col] = pd.to_numeric(cv[col])

        return cv


if __name__ == '__main__':
    df = PaperFactory.entries_to_df(PaperFactory.many_from_json('../data/papers.json'))

    df.info()
