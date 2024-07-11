import torch
from transformers import pipeline #テキスト解析
import speech_recognition as sr #音声認識
from moviepy.editor import VideoFileClip #動画と音声を再生
import pygame #ビデオウィンドウの管理と表示に使用

# 音声認識のセットアップ
recognizer = sr.Recognizer() #Recognizerおbジェクトを作成。音声認識に使用。

# テキスト解析モデルのロード
nlp = pipeline('sentiment-analysis')

# 音声付き動画再生関数
def play_video_with_sound(file_path):
    # Pygame初期化
    pygame.init()
    pygame.display.set_caption("Video") #ウィンドウのタイトルをVideoに設定。
    
    # ビデオクリップのロード
    clip = VideoFileClip(file_path)

    # Pygameディスプレイのセットアップ
    screen = pygame.display.set_mode((clip.w, clip.h)) #Pygameのディスプレイを動画の幅と高さに設定。

    # 動画再生
    clip.preview()

    # Pygameの終了
    pygame.quit()

# 音声を認識し、テキスト解析を行う関数
def recognize_and_respond():
    with sr.Microphone() as source: #マイクを音声入力のソースとして使用
        print("Please speak:") #
        audio = recognizer.listen(source) #マイクから音声を取得し、audioに保存
        
        try:
            text = recognizer.recognize_google(audio, language='ja-JP') #Googleの音声認識APIを使用して音声をテキストに変換
            print(f"You said: {text}")
            
            if "厳しいって" in text:
                print("Keyword detected! Playing video...")
                play_video_with_sound('assets/kibishii.mp4')  # ここにMP4ファイルのパスを指定
            else:
                print("Keyword not detected.")
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results; {e}")

# メイン関数
if __name__ == "__main__": #スクリプトが直接実行された場合に以下のコードを実行。
    while True:
        recognize_and_respond()
