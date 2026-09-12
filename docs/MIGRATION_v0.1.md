# Migration to PoC v0.1 layout

現在のリポジトリ直下にあるスクリプトを、次のように整理します。

## 本線 → src/

- `download_gsi_geotiff.py`
- `predict_geotiff_tiled.py`
- `sieve_landcover.py`
- `polygonize_landcover.py`
- `analyze_landcover_gpkg.py`

## 初期検証 → legacy/

- `check_rgb_model.py`
- `download_gsi_image.py`
- `predict_rgb.py`
- `predict_rgb_tiled.py`

## 直下に残す

- `run_poc.py`
- `README.md`
- `requirements.txt`
- `LICENSE`
- `.gitignore`

`run_poc.py` は `src/` 内の本線コードを呼ぶように更新済みです。

## 移行後の動作確認

```bat
python run_poc.py --lat 35.662 --lon 140.070 --name layout_test --polygonize-raw
```

前回と同じ構成の `runs/...` が生成されれば移行成功です。

## 注意

以前のコマンドで直接

```bat
python predict_geotiff_tiled.py ...
```

としていた場合、移行後は

```bat
python src\\predict_geotiff_tiled.py ...
```

になります。

通常は `run_poc.py` を標準入口として使用してください。
