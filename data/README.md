# Data directory

Large assets are intentionally excluded from Git.

- `touchdown/`: navigation JSON files, graph files, and rule task manifests.
- `features/`: base and inserted FLAME feature LMDBs.
- `images/`: reconstructed 120-degree raw-image LMDBs used by SNA.
- `official_touchdown/`: user-provided official StreetLearn/Touchdown release.
- `citynav_rules/`: CityNav-Rules images obtained under their source terms.

Run `python scripts/download_assets.py` for released Rule-VLN assets and
`python scripts/prepare_touchdown.py` for the official graph and panoramas.
