# prompt_ops

## dspy + ColBERT

- setup env with pixi
- clone ColBERT
- create index using notebook
- copy experiments from notebook dir to ColBERT/experiments
- `pixi run colbert_server`

## ColBERT

- [x] env setup
- [x] FastAPI
- [x] class for wrapping Searcher in app
- [x] search config in `colbert_server.py` (moved from ColBERT/.env)
- [x] setup step for ColBERT repo
- [ ] parametrize main in `colbert_server.py` with the config
- [ ] script to build the index
- [ ] pixi commands for running the app with parameters and indexing
- [ ] configure indexing to provide the collection, for now the collection is downloaded separately in colbert_server

## dspy

- [x] env setup
- [x] script to run dspy with remote ColBERT
- [x] run dspy using local ColBERT
- [ ] create a server
