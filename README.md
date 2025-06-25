
# 日本語ゲームセリフ 感情ラベリングプロジェクト

このプロジェクトは、日本語のゲームセリフの音声ファイルとテキストのペアに対して、自動で感情ラベルを付与するためのツールです。

音声認識には `microsoft/wavlm-base-plus-jp-sentiment` モデルを、テキスト分析には `Google Cloud Natural Language API` を使用したハイブリッドアプローチを採用しています。

## 1. セットアップ

このプロジェクトは、Python 3.11以上と `uv` パッケージマネージャーを前提としています。

### a. Google Cloudの認証設定

本ツールは Google Cloud Natural Language API を使用します。

1.  Google Cloud Platformでプロジェクトを作成し、**Natural Language API** を有効にしてください。
2.  サービスアカウントを作成し、キー（JSONファイル）をダウンロードします。
3.  ダウンロードしたキーファイルへのパスを環境変数 `GOOGLE_APPLICATION_CREDENTIALS` に設定します。

    ```shell
    # Windows (コマンドプロンプト)
    setx GOOGLE_APPLICATION_CREDENTIALS "C:\path\to\your\keyfile.json"

    # Windows (PowerShell)
    $env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\your\keyfile.json"
    ```
    **注意:** コマンドプロンプトを再起動しないと環境変数が反映されない場合があります。

### b. Python仮想環境の作成とライブラリのインストール

プロジェクトのルートディレクトリで、以下のコマンドを実行します。

**【重要】Pythonのバージョンについて**
本プロジェクトの依存ライブラリ(`librosa`)の都合上、**Python 3.10** または **Python 3.11** の利用を推奨します。
(Python 3.12以上では、`numba` の依存関係解決で問題が発生する可能性があります。)

```shell
# 1. uvで仮想環境を作成 (例: Python 3.11を使う場合)
# uv venv --python 3.11
uv venv

# 2. 仮想環境を有効化 (Windows)
.venv\Scripts\activate

# 3. 必要なライブラリをインストール
uv pip install -r requirements.txt
```

**トラブルシューティング:**
`uv pip install` で `torch` や `numba` に関するエラーが発生した場合、`requirements.txt` 内の `torch` の行から `--index-url` 以降を削除して、`torch` のみ記述した状態でお試しください。

## 2. データ準備

### a. 音声ファイル

全ての音声ファイル（.wav, .mp3など）を一つのディレクトリにまとめてください。

### b. メタデータファイル

音声ファイル名とセリフのテキストを対応付けた `metadata.csv` を作成します。
パイプ区切り(`|`)で、ヘッダーは `audio_filename` と `text` にしてください。

**例: `metadata.csv`**
```csv
audio_filename|text
dialogue_001.wav|全軍、続け！私に続け！
dialogue_002.wav|どうして…こんなことに…。
dialogue_003.wav|よくも仲間を！許さん！
```

## 3. 実行方法

仮想環境を有効化した状態で、以下のコマンドを実行します。

```shell
python main.py [入力CSVのパス] [音声ディレクトリのパス] [出力CSVのパス]
```

**実行例:**

```shell
python main.py C:\Users\yuta\Desktop\data\metadata.csv C:\Users\yuta\Desktop\data\wavs C:\Users\yuta\Desktop\data\metadata_labeled.csv
```

実行が完了すると、指定した出力パスに `emotion_label` 列が追加された新しいCSVファイルが生成されます。

## 4. 統合ロジックのカスタマイズ

本ツールの感情判定の精度は `main.py` 内の `_integrate_results` メソッドに大きく依存します。

```python
def _integrate_results(self, audio_emotion: str, text_sentiment: dict) -> str:
    # ... (ここのロジックを調整)
```

データセットの特性（例: 皮肉が多い、特定の口癖があるなど）に合わせてこのロジックを調整することで、ラベリングの精度をさらに向上させることができます。
