# prompt_ops

## dspy + ColBERT

- setup env with pixi
- clone ColBERT
- create index using notebook
- copy experiments from notebook dir to ColBERT/experiments
- `pixi run colbert_server`

##

- [x] FastAPI
- [x] class for wrapping Searcher in app
- [ ] search config in `colbert_server.py` (moved from ColBERT/.env)
- [ ] configurable main in `colbert_server.py`
- [ ] setup step for ColBERT repo
- [ ] script to build the index
- [ ] pixi commands for running the app with parameters and indexing
- [ ] configure indexing to provide the collection, for now the collection is downloaded separately in colbert_server
