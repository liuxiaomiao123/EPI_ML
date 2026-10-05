
import pandas as pd

def encode_percentiles(pro):
    return pd.qcut(pro, q=[0, 0.25, 0.75, 1], labels=['low', 'medium', 'high'])


def normalize_row(row):
    out = {}
    for short, col in FIELD_MAP.items():
        val = row[col]
        out[short] = 'NA' if pd.isna(val) else val
    return out


class Registry:
    def __init__(self):
        self.items = {}      
        self.meta = {}       
        self.pairs = set()   

    def put(self, kind, **fields):
        key = (kind, tuple(sorted(fields.items())))
        if key not in self.items:
            idx = len(self.items)
            self.items[key] = idx
            self.meta[idx] = {'kind': kind, **fields}
        return self.items[key]

    def pair(self, a, b, code):
        self.pairs.add((a, b, code))

    def size(self):
        return len(self.items), len(self.pairs)


def make_subject(reg, rec):
    return reg.put('P', id=rec['id'], age=rec['age'], gender=rec['sex'])


def make_markers(reg, rec):
    return {
        'p1': reg.put('A1', name='pro1', level=rec['a1']),
        'p2': reg.put('A2', name='pro2', level=rec['a2']),
        'p3': reg.put('A3', name='pro3', level=rec['a3']),
    }


def make_stage(reg, rec):
    return reg.put('S', stage=rec['stg'])


def make_type(reg, rec):
    return reg.put('H', type=rec['his'])


def make_status(reg, rec):
    return {
        'er': reg.put('E', status=rec['er']),
        'pr': reg.put('R', status=rec['pr']),
        'c2': reg.put('C', status=rec['c2']),
    }


def collect_ids(reg, rec):
    ids = {'pt': make_subject(reg, rec)}
    ids.update(make_markers(reg, rec))
    ids['st'] = make_stage(reg, rec)
    ids['hi'] = make_type(reg, rec)
    ids.update(make_status(reg, rec))
    return ids


MARKERS = ('p1', 'p2', 'p3')
STATUSES = (('er', 3), ('pr', 4), ('c2', 5))


def layer_one(ids, reg):
    for m in MARKERS:
        reg.pair(ids['pt'], ids[m], 0)


def layer_two(ids, reg):
    reg.pair(ids['pt'], ids['st'], 1)
    for m in MARKERS:
        reg.pair(ids['st'], ids[m], 0)


def layer_three(ids, reg):
    reg.pair(ids['pt'], ids['hi'], 2)
    reg.pair(ids['st'], ids['hi'], 2)
    for left in ('pt', 'st', 'hi'):
        for key, code in STATUSES:
            reg.pair(ids[left], ids[key], code)


LAYERS = {1: layer_one, 2: layer_two, 3: layer_three}

def ingest(reg, row, depth=3):
    rec = normalize_row(row)
    ids = collect_ids(reg, rec)
    for d in range(1, depth + 1):
        LAYERS[d](ids, reg)


def run_all(df, depth=3):
    reg = Registry()
    for _, row in df.iterrows():
        ingest(reg, row, depth=depth)
    return reg


def to_tables(reg):
    items = pd.DataFrame.from_dict(reg.meta, orient='index')
    items.index.name = 'idx'
    cooc = pd.DataFrame(sorted(reg.pairs), columns=['a', 'b', 'code'])
    return items, cooc
