# SACLAJ Evaluation v0.1

Phase A Minimal End-to-Endの評価部分です。Base model → GSI paddy class-7 positive-only
partial-label fine-tuning済みcheckpoint → SACLAJ independent point reference →
Base / Fine-tuned paired comparisonを行います。再学習やproduction accuracyの証明はしません。

## Category_ID mappingとprovenance

SACLAJ README・実CSVと照合した利用者レビューに基づき、v0.1 mappingを以下で確定しました。
`Category_ID`が正式な土地被覆カテゴリです。`Category_detail`は空欄または自由記述の補足情報で、
**採否・mapping・samplingの判定には一切使用しません**。元のdetailはlocal結果だけに保持します。
[JAXA公式配布案内](https://www.eorc.jaxa.jp/ALOS/jp/dataset/lulc_j.htm)の
SACLAJ Reference Dataset ver.25.06を対象にします。

`saclaj_mapping.template.json`には確定した14個のCategory_ID（採用10・明示除外4）を同梱します。
会社PCでリポジトリ外へコピーし、配布READMEファイルの`definition_sha256`を記入してください。
READMEファイル本体はこの開発環境にないためhashを捏造せず空欄としています。
`definition_verified=true`は利用者による原典照合を反映しますが、hash未記入ではfail-fastします。
実データによるEnd-to-End評価は会社PCでの実行事項です。

JSONは次の仕様です。旧schema_version=1とCategory_detail付きentriesは受け付けません。

- `schema_version`: 2（Category_ID単位）。
- `mapping_version`: 確認したdatasetと対応表の版を識別する文字列。
- `definition_source`: 正式READMEの文書名・版・節など。絶対pathは不要です。
- `definition_sha256`: 確認したREADMEファイルそのもののSHA256、小文字64桁。
- `definition_verified`: 人が原典との照合を終えた場合だけtrue。ソフトウェアが原典の正しさを証明するものではありません。
- `entries`: `category_id`（非負整数の文字列）と`subtype`（次表の値または除外を示すnull）だけ。

Category_IDの重複はエラーです。未登録IDとnull指定IDはdetailの内容によらず除外します。
mappingには少なくともriceと非Agricultureの採用カテゴリが必要です。

| Category_ID | 評価subtype | OEM8 class |
|---:|---|---|
| 9 | rice_paddy | 7 Agriculture |
| 11 | other_crop | 7 Agriculture |
| 18 | mixed | 5 Tree |
| 19 | needleleaf | 5 Tree |
| 20 | broadleaf | 5 Tree |
| 22 | needleleaf_evergreen | 5 Tree |
| 23 | broadleaf_evergreen | 5 Tree |
| 27 | needleleaf_deciduous | 5 Tree |
| 28 | broadleaf_deciduous | 5 Tree |
| 33 | water | 6 Water |

8 croplands、10 pasture、17 forest、32 urban and built-up、その他未登録IDは除外。
urban and built-upをBuildingへ対応させません。22/23/27/28は19/20へ統合せず、
各subtypeを独立して最大N点抽出・集計します。OEM8 class番号は`src/config.py`と同じです。

## Referenceと空間・時間の限界

SACLAJは今回のfine-tuningの教師に使わない独立したpoint referenceです。ただし、
GSI training領域との地理的重複やBase modelのpretrainingとの重複を検証した意味での
統計的な独立性は主張しません。pixel-perfect GTでもありません。

日本域はWGS84の **122°E ≤ longitude ≤ 154°E、20°N ≤ latitude ≤ 46°N** を使います。
境界を含みます。国境polygonではないため、範囲内の国外領域・海域も含まれます。
この限界と範囲はmanifest / aggregate summaryにも入ります。

**SACLAJ観測時期とGSI最新写真の撮影時期が異なる可能性があり、実際の土地被覆変化が
evaluation mismatchとして現れ得ます。** 過去写真探索や厳密temporal matchingは行いません。
取得日時は撮影日時ではありません。`Date`はlocal結果に保持し、撮影日の代用にしません。

`Diameter（m）`は記録情報だけです。GT circle / polygonやneighborhood majorityには使いません。
GPS誤差、細い地物、境界画素、画像位置ずれ、季節変化、カテゴリの意味の差でagreementが
下がる可能性があります。将来のDCHMやtree-species評価に向け、森林subtypeを混合しません。

## CSV仕様とpreflight

既定はUTF-8（BOM可）、カンマ区切り、引用符を使う通常のCSVです。
列名は次の8個が必要です。末尾の空名の列を含む追加列は無視します。

`ID, Category_ID, Latitude, Longitude, Date, Diameter（m）, Category_detail, Note`

- Diameterは全角括弧の`Diameter（m）`が正式名、ASCIIの`Diameter(m)`もaliasとして許可。
  どちらか1列が必須で、両方ある場合は値が同じでも曖昧なschemaとして停止します。
  manifestの`csv_resolved_columns.diameter`に実際の列名を記録します。
  local site_results側の出力名は従来の`Diameter(m)`を維持します。
- ヘッダ重複・必須列欠落・行の列数不一致は停止。
- IDは周囲空白除去後に空でない一意の値。同じ地点の複数観測で同一IDが繰り返されるCSVなら
  停止するので、正式なレコード識別仕様を確認してadapterを修正してください。
- Category_IDは非負の整数文字列。`3.0`などを勝手に整数化しません。先頭ゼロは正規化します。
- Latitude / LongitudeはWGS84十進度、有限数、緯度[-90,90]・経度[-180,180]。
- Dateは空でない文字列。書式・実在日・時期は検証せず、 temporal matchingに使いません。
- Diameterは空欄または有限の非負数。Category_detail / Noteは空欄可。
- Noteや追加列は保持せず、集計・manifestに任意の入力文章をコピーしません。
- 文字コードが異なる場合だけ`--csv-encoding cp932`等を明示します。自動推測しません。

利用者レビューでは実CSVは63,968件、ID全件一意、Date空欄なしです。
この件数をコードに固定せず、毎runでschemaと件数を検査します。実CSV本体をこの開発環境では
読み込んでいないため、会社PCでpreflightを実行してください。

## Pipelineと再現性

1. 入出力先がGit checkout外であることを検証し、確認済みmappingとCSV全行を検証。
2. 日本域bbox → 明示mappingの順にfilterし、段階ごとの総数と採用subtype数を記録。
3. 各subtypeの`[seed, subtype, ID]`を空白なしUTF-8 JSONにしてSHA256を計算し、昇順で先頭N点を抽出。
   tieはID昇順。既定N=100、seed=42。同じ入力・filter・seed・NならCSV行順によらず同じ集合です。
   画像取得前に抽出し、失敗した地点を別地点で補充しません。
4. CLIで指定したBase / FTを`src/model.py::build_model`でstrict load。 architectureは
   SMP Unet / EfficientNet-B4 / scSE、3入力・9出力。0 Backgroundを含む既存class IDを保持。
5. 共通loaderと512×512ゼロRGBで両forwardをoffline検査。preflightならここで終了。
6. 各地点で既存`lonlat_to_tile`と`download_tile`を再利用し、GSI全国最新写真seamlessphotoの
   XYZタイルを取得。既定zoom=18（CLIで14–18）。timeoutはタイル当たり30秒、retryなし。
7. global XYZ pixel座標をfloorし、その地点を含む画素をpatchの(row=256, col=256)へ配置。
   512×512 RGBを切り出し、必要な最大3×3タイルを全て取得。resize / resampling / 黒埋めなし。
   固定zoomなので地上の実寸は緯度に依存します。lossless PNGとSHA256をlocal保存。
8. `src/model.py::rgb_to_tensor`でuint8 RGB → float32 /255 → CHW。
   mean/std normalizationなし。同一tensorのcopyをそれぞれのeval modelへ入力し、
   9 logitsの中心1画素にsoftmax / argmax。近傍投票・sieve・tiled overlap平均は行いません。
9. BaseとFTの両predictionが成功した同じ地点集合だけでpaired集計。

404/410は`skipped/imagery_unavailable`、その他HTTP・network・decode失敗は`error`として
地点行と件数を保存します。結果を捏造せず両モデルの指標から除外します。
モデルforward失敗は当該地点をerrorにしてrunを停止。完了済み地点とfailed manifestを保持します。
全取得失敗またはrice/non-Agricultureの成功ペア欠落は`insufficient_pairs`、exit code=2。
正常完了/preflightは0、例外は1。母数ゼロの指標は0%ではなくnullです。
強制終了・電源断の場合はmanifestがrunningのまま残ることがあります。

preflightはネットワーク疎通や実画像品質を検証しません。同一取得画像を比較しても、
別runでは最新GSI写真が更新される場合があります。local保存patchが評価時の入力です。
全データをGPUに保持せず1地点ずつ処理しますが、モデル2個を同時に保持します。

## Metrics

以下の%指標はそれぞれの成功ペアを分母nに取り、delta_pp = FT% − Base%です。

- **Rice agreement**: rice_paddyでclass 7を予測した割合。
- **Agriculture leakage**: expected class ≠7（採用Tree / Water）の地点でclass 7を予測した割合。
  riceとother_cropを分母に含めません。positive-only学習によるclass-7過剰予測/collapseを確認する
  重要指標で、riceの改善だけで成功を判断しません。leakageの増加は悪化方向です。
- **Other-class retention**: 全7種の森林subtypeとwaterを別々に、expected classとの一致率。
  Baseで正解だった点だけを分母にする条件付き指標ではなく、各subtypeの全成功ペアで比較します。
- **Probability shift**: 各subtypeのexpected-class確率とAgriculture確率の平均、FT−Baseの差。
  確率とdeltaは0–1単位（deltaは[-1,1]）で、percentage pointsとは区別します。

単純なoverall accuracyや精度改善宣言は出しません。カテゴリ別上限抽出は日本全体の
クラス頻度を反映せず、少数点・欠測・カテゴリ対応の限界もあるためです。

## Outputsと情報管理

```text
<output-dir>/<UTC timestamp + unique suffix>/
├── .gitignore                  # 内容すべてを除外
├── evaluation_manifest.json
├── site_results.csv            # 会社PCローカル限定
├── patches/000001.png          # 会社PCローカル限定
└── aggregate_summary.json      # 座標・ID・地点別結果を含まないsanitized aggregate
```

CSV入力とoutput-dirをリポジトリ内・別のGit checkout配下に指定すると拒否します。
各runに`.gitignore`を自動作成し、repoのignoreにもSACLAJ CSVと結果名を追加しています。
`git add -f`等の強制操作やGit管理開始を防ぐアクセス制御ではありません。
**SACLAJ CSV・座標・地点ID・地点別結果・patch・実行結果をGitHubや外部環境へ追加しないでください。**

site_resultsにはID、座標、元Category_ID/detail、Date、Diameter、expected class、
各modelのpredicted class / expected probability / Agriculture probability、
skip/error reason、patch参照/SHA256・取得日時が入ります。利用は会社PCローカルのみ。

manifestにはrun ID/UTC、Base・FT・CSV・mappingのSHA256、mapping版と原典SHA256、
bbox、seed/N、filter前後/抽出数、GSI source・取得仕様、共通architecture/preprocessing、
versions/device、解決したDiameter列名、成功/skip/error/未処理数、出力ファイル名、既知の限界を記録します。
CSV絶対path・地点ID・地点座標を転記しません。manifestもlocal実行結果として扱います。

aggregateはallowlistから新たに組み立て、任意のCSV本文・例外文・pathをコピーしません。
input/filter/sample counts、固定subtype、metrics/delta、skip/error、bbox定義と限界を保持します。
**「座標・ID・地点別結果を含まないsanitized aggregate」であり、公開可とコードは判定しません。**
Git管理対象はコード・tests・docs・集計schemaのみ。集計JSON自体も自動commit対象にしません。

## Windows cmdでの最初の実行

Python 3.11とPhase Aで使ったvenvを使用します。FTには1-epoch Pilotのbest.pthを指定します。
pathは全てCLI引数で、コードには固定していません。

同梱mapping templateをローカルへコピーし、README hashを記入します。定義ファイルのhashは例えば
`certutil -hashfile C:\OpenEarthMap_PoC\data\saclaj\raw\README.txt SHA256`で確認できます。
実際のファイル名を指定し、出力hashを小文字にしてmappingへ記載してください。

以下のsetの値を実際の保存先に変更してから、最初にpreflightを実行してください。

```bat
cd /d C:\OpenEarthMap_PoC\openearthmap-poc
call .venv\Scripts\activate.bat
set "SACLAJ_CSV=C:\OpenEarthMap_PoC\data\saclaj\raw\Gref_DB_2025_06.csv"
set "MAPPING=C:\OpenEarthMap_PoC\data\saclaj\saclaj_mapping.v0.1.json"
set "BASE=C:\OpenEarthMap_PoC\models\RGB_Real_5_u-efficientnet-b4.pth"
set "FT=C:\OpenEarthMap_PoC\training_outputs\gsi_phase_a\YOUR_1EPOCH_RUN\checkpoints\best.pth"
set "RESULTS=C:\OpenEarthMap_PoC\data\saclaj\results"

python -m src.evaluation.evaluate_saclaj ^
  --saclaj-csv "%SACLAJ_CSV%" --mapping "%MAPPING%" ^
  --base-model "%BASE%" --fine-tuned-model "%FT%" ^
  --output-dir "%RESULTS%" --device cpu --num-threads 2 ^
  --max-samples-per-category 100 --seed 42 --preflight
```

preflight成功後、同じ引数から`--preflight`を外して実評価を開始します。

```bat
python -m src.evaluation.evaluate_saclaj ^
  --saclaj-csv "%SACLAJ_CSV%" --mapping "%MAPPING%" ^
  --base-model "%BASE%" --fine-tuned-model "%FT%" ^
  --output-dir "%RESULTS%" --device cpu --num-threads 2 ^
  --max-samples-per-category 100 --seed 42
```

mappingとCSV仕様は利用者レビューを反映済みです。README実ファイルのhash、実Base/FT互換性、
実GSI取得、会社PCのRAM/時間は会社PCで確認してください。
実画像の取得ではXYZタイル要求がGSIへ送られますが、SACLAJ CSV・ID・観測内容は送信しません。
実SACLAJデータをこの開発環境へ持ち込まず、会社PCで実行してください。

## Offline検証

```bat
python -m pytest -q
python -m compileall -q run_poc.py src tests
```

合成CSV / 確定したカテゴリ定義 / dummy model / mocked GSIだけを使用します。
地点ID・座標・補足情報・provenance hashは架空のtest値で、実データfixtureは保存しません。
schema検証、曖昧カテゴリ除外、日本bbox、決定的subtype抽出、上限、paired集計、rice、leakage、
森林subtype、確率差、取得失敗、sanitization、manifest、共通loader/preprocessing、
tile境界でのpoint位置、offline preflight、全失敗時のnull、全角/ASCII Diameter、
解決列名manifest、空名追加列、detailに依存しない採否、常緑/落葉subtypeの独立集計を検証します。

TrainingへのSACLAJ利用、GT circle、近傍評価、過去写真探索、DCHM、tree-species polygons、
MAFF、GSI conifer teacher、manual GT、distillation、pseudo-labeling、GUI/exe、
hyperparameter search、追加再学習は範囲外です。
