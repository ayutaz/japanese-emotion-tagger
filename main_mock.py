# -*- coding: utf-8 -*-

import torch
import librosa
import pandas as pd
from transformers import pipeline
from google.cloud import language_v1
import os
import argparse

class JapaneseEmotionTagger:
    """
    日本語の音声とテキストから感情を判定するハイブリッド・パイプライン。
    - 音声分析: Microsoft/WavLMベースのモデルを使用
    - テキスト分析: Google Cloud Natural Language APIを使用
    """

    def __init__(self):
        """モデルとクライアントを初期化 (モック)"""
        print("モデルの初期化をスキップします (モックモード)")
        # 実際のモデルロードとクライアント初期化は行わない
        self.audio_pipeline = None
        self.gcp_language_client = None
        print("初期化が完了しました。")

    def analyze_audio(self, audio_path: str) -> str:
        """
        音声ファイルから感情を分析する (モック)。
        """
        print(f"音声分析をスキップ: {os.path.basename(audio_path)}")
        if "001" in audio_path:
            return "happy"
        else:
            return "normal"

    def analyze_text(self, text: str) -> dict:
        """
        テキストから感情を分析する (モック)。
        """
        print(f"テキスト分析をスキップ: '{text}'")
        if "テスト用" in text:
            return {"score": 0.8, "magnitude": 1.2}
        else:
            return {"score": -0.5, "magnitude": 0.8}

    def _integrate_results(self, audio_emotion: str, text_sentiment: dict) -> str:
        """
        音声とテキストの分析結果を統合し、最終的なラベルを決定する。
        """
        text_score = text_sentiment['score']
        
        if audio_emotion == 'angry' and text_score < -0.3:
            return 'angry'
        if audio_emotion == 'happy' and text_score > 0.3:
            return 'happy'
        if audio_emotion == 'sad' and text_score < -0.3:
            return 'sad'

        if audio_emotion in ['angry', 'happy', 'sad']:
            return audio_emotion
            
        if text_score > 0.7:
            return 'very_happy'
        if text_score < -0.7:
            return 'very_sad_or_angry'

        if audio_emotion == 'normal':
            return 'neutral'
        
        return 'neutral'

    def tag_emotion(self, audio_path: str, text: str) -> str:
        """
        単一の音声・テキストペアに対して、最終的な感情ラベルを付与する。
        """
        print(f"--- 処理中: {os.path.basename(audio_path)} ---")
        audio_emotion = self.analyze_audio(audio_path)
        text_sentiment = self.analyze_text(text)
        final_label = self._integrate_results(audio_emotion, text_sentiment)
        print(f"  音声: {audio_emotion}, テキスト: score={text_sentiment['score']:.2f} -> 最終ラベル: {final_label}")
        return final_label

def process_metadata(tagger: JapaneseEmotionTagger, input_csv: str, audio_dir: str, output_csv: str):
    """
    メタデータCSVを処理して、感情ラベルを付与する。
    """
    try:
        df = pd.read_csv(input_csv, sep='|')
    except FileNotFoundError:
        print(f"[エラー] 入力CSVファイルが見つかりません: {input_csv}")
        return

    df['emotion_label'] = ''

    for index, row in df.iterrows():
        # audio_dir とファイル名を結合して完全なパスを作成
        audio_path = os.path.join(audio_dir, row['audio_filename'])
        
        if not os.path.exists(audio_path):
            print(f"[警告] 音声ファイルが見つかりません: {audio_path}。スキップします。")
            df.at[index, 'emotion_label'] = 'file_not_found'
            continue

        text = row['text']
        label = tagger.tag_emotion(audio_path, text)
        df.at[index, 'emotion_label'] = label

    df.to_csv(output_csv, sep='|', index=False)
    print(f"\n処理が完了し、'{output_csv}' に保存しました。")

def main():
    parser = argparse.ArgumentParser(description="日本語音声・テキストデータセットに感情ラベルを付与します。")
    parser.add_argument("input_csv", help=r"入力メタデータCSVファイルのパス (例: metadata.csv)")
    parser.add_argument("audio_dir", help=r"音声ファイルが格納されているディレクトリのパス (例: C:\Users\yuta\Desktop\Private\wav_files)")
    parser.add_argument("output_csv", help=r"感情ラベルを付与したメタデータCSVの出力パス (例: metadata_labeled.csv)")
    
    args = parser.parse_args()

    # Google Cloudの認証情報が設定されているか確認 (モックのため無効化)
    # if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
    #     print("[エラー] 環境変数 'GOOGLE_APPLICATION_CREDENTIALS' が設定されていません。")
    #     print("GCPのサービスアカウントキー(JSON)へのパスを設定してください。")
    #     return

    tagger = JapaneseEmotionTagger()
    process_metadata(tagger, args.input_csv, args.audio_dir, args.output_csv)

if __name__ == '__main__':
    main()