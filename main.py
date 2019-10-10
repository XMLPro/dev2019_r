import cv2
import numpy as np
import pygame
from numpy.random import *


# フレーム差分の計算
def frame_sub(img1, img2, img3, th):
    # フレームの絶対差分
    diff1 = cv2.absdiff(img1, img2)
    diff2 = cv2.absdiff(img2, img3)

    # 2つの差分画像の論理積
    diff = cv2.bitwise_and(diff1, diff2)

    # 二値化処理
    diff[diff < th] = 0
    diff[diff >= th] = 255
    
    # メディアンフィルタ処理（ゴマ塩ノイズ除去）
    mask = cv2.medianBlur(diff, 5)

    return diff

def alarm():
    soundfile_1 = "punch-swing1.mp3"
    soundfile_2 = "b2-026_swing_02.mp3"
    soundfile_3 = "punch-swing1.wav"
    pygame.mixer.init(frequency = 44100)    # 初期設定
    a = rand()/2 + rand()/2
    if a < 0.3:
        pygame.mixer.music.load(soundfile_2)
    elif a >= 0.3 and a < 0.7:
        pygame.mixer.music.load(soundfile_1)
    else:
        pygame.mixer.music.load(soundfile_3)

    #pygame.mixer.music.load(soundfile_1)

    pygame.mixer.music.play(1)
    return 0

def textWindow(min_moment, EffectForever):
    img = np.zeros((200,512,3), np.uint8)
    text1 = "Effect"
    text2 = "MoveJudgeNum"
    text3 = "On"
    text4 = "Off"
    cv2.putText(img, text1, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2, cv2.LINE_AA)
    cv2.putText(img, text2, (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2, cv2.LINE_AA)

    if EffectForever == False:
        cv2.putText(img, text3, (250,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2, cv2.LINE_AA)
    else:
        cv2.putText(img, text4, (250,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2, cv2.LINE_AA)
    cv2.putText(img, str(min_moment), (250,100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2, cv2.LINE_AA)
    
    cv2.putText(img, "Effect : s, MoveJudgeNum : a", (20,150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv2.LINE_AA)
    
    cv2.imshow("text", img)
    

def main():

    # 判定の閾値
    #PC前近く10000
    #ちょっと遠く　5000前後
    min_moment = 6000

    # カメラのキャプチャ
    cap = cv2.VideoCapture(0)
    
    # フレームを3枚取得してグレースケール変換
    frame1 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
    frame2 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
    frame3 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
    
    # カウント変数の初期化
    cnt = 0
    count = 6

    #音声発生時確認用Flag
    swingFlag = False
    effectFlag = False
    EffectForever = False

    while(cap.isOpened()):
        ret, frame = cap.read()
        # フレーム間差分を計算
        mask = frame_sub(frame1, frame2, frame3, th=20)
        ## +メディアンフィルタ処理（ゴマ塩ノイズ除去）
        mask_m = cv2.medianBlur(mask, 5)
        # 白色領域のピクセル数を算出
        moment = cv2.countNonZero(mask_m)

        # グレースケールに変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



        if moment > min_moment and count > 2:
            swingFlag = True
            count = 0
        elif moment < min_moment:
            if count < 10:
                count += 1
            if count > 8:
                effectFlag = False



        if swingFlag == True:
            alarm()
            effectFlag = True
            swingFlag = False
        
        #エッジ抽出
        #canny = cv2.Canny(mask_m, 50, 110)
        canny = cv2.Canny(mask_m, 20, 140)
        
        fframe = cv2.split(frame)

        if effectFlag == True and EffectForever == False:
            fframe[0] += canny * 50
            fframe[1] += canny * 50
            fframe[2] += canny * 30


        ffframe = cv2.merge(fframe)

        # 結果を表示
        #cv2.imshow("Mask", mask_m)
        #cv2.imshow("capture", cv2.flip(frame, 1))
        #cv2.imshow("flip", cv2.flip(mask_m, 1))
        cv2.imshow("blend", cv2.flip(ffframe, 1))
        #cv2.imshow("edge", cv2.flip(canny, 1))
        #cv2.imshow("blend", cv2.flip(blended, 1))

        # 3枚のフレームを更新
        frame1 = frame2
        frame2 = frame3
        frame3 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
        
        #情報表示用ウィンドウ作成
        textWindow(min_moment, EffectForever)
        
        # qキーが押されたら途中終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(1) & 0xFF == ord('a'):
            min_moment += 1000
            if min_moment > 10000:
                min_moment = 5000
        elif cv2.waitKey(1) & 0xFF == ord('s'):
            if EffectForever == True:
                EffectForever = False
            else:
                EffectForever = True


    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
