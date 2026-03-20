CONFIGS = {
    'roman-empire':   dict(d=256, K=8,  order=4, layers=2, lr=3e-4, wd=1e-4, epochs=1500, dropout=0.4),
    'questions':      dict(d=256, K=8,  order=4, layers=2, lr=3e-4, wd=1e-4, epochs=500,  dropout=0.4),
    'amazon-ratings': dict(d=256, K=6,  order=4, layers=2, lr=3e-4, wd=1e-4, epochs=500,  dropout=0.4),
    'tolokers':       dict(d=256, K=6,  order=3, layers=2, lr=5e-4, wd=1e-4, epochs=750,  dropout=0.3),
    'minesweeper':    dict(d=256, K=4,  order=4, layers=2, lr=5e-4, wd=1e-4, epochs=3000, dropout=0.3),
    'cora':           dict(d=256, K=3,  order=2, layers=1, lr=5e-4, wd=1e-3, epochs=1000, dropout=0.5),
    'cs':             dict(d=256, K=6,  order=4, layers=2, lr=3e-4, wd=1e-4, epochs=3000, dropout=0.3),
    'coauthor-cs':    dict(d=256, K=6,  order=3, layers=2, lr=3e-4, wd=1e-4, epochs=400,  dropout=0.3),
    'amazon-photo':   dict(d=256, K=6,  order=4, layers=2, lr=3e-4, wd=1e-4, epochs=3000, dropout=0.3),
    'photo':          dict(d=256, K=6,  order=4, layers=2, lr=3e-4, wd=1e-4, epochs=400,  dropout=0.3),
    'wikics':         dict(d=256, K=6,  order=4, layers=2, lr=3e-4, wd=1e-4, epochs=500,  dropout=0.4),
    'wiki-cs':        dict(d=256, K=6,  order=4, layers=2, lr=3e-4, wd=1e-4, epochs=500,  dropout=0.4),
}

DEFAULT_CFG = dict(d=256, K=3, order=4, layers=2, lr=3e-4, wd=1e-4, epochs=3000, dropout=0.4)

N_SPLITS = {
    'roman-empire': 10, 'questions': 10, 'amazon-ratings': 10,
    'tolokers': 10, 'minesweeper': 10,
    'cora': 1, 'citeseer': 1, 'pubmed': 1,
    'wikics': 20, 'wiki-cs': 20,
}

DEFAULT_N_SPLITS = 5

VALID_DATASETS = [
    'cora', 'cs', 'amazon-photo',
    'roman-empire', 'tolokers', 'amazon-ratings',
    'minesweeper', 'questions', 'citeseer',
    'amazon-computers', 'physics', 'wikics', 'all',
]
