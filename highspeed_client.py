import numpy as np
import cv2
import requests
import cv2
import base64
from io import BytesIO

# remote_address = "157.82.204.56"
remote_address = "macbookpro-hayaken.local"

cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FPS, 90)           # カメラFPSを60FPSに設定
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # カメラ画像の横幅を1280に設定
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
fgbg = cv2.createBackgroundSubtractorMOG2(0,7)

margin = 2
rangethreshold = 10

detected_frame = []
count = 0

START_TIME = 4 
TIMEOUT = 10
MAXTIME = 10
mode = 0 # 0:なにもない, 1:写っている
current_time = 0
FRAME_W = 320
FRAME_H = 240
# default = np.zeros(FRAME_W,FRAME_H,3)
flag_test = True

def send_image(img):
    print(img.shape)
    # img = img.resize((100,180))
    is_success, buffer = cv2.imencode(".jpg", img)
    io_buf = BytesIO(buffer)
    b64_img = base64.b64encode(io_buf.getvalue()).decode()
    payload = {'width':img.shape[0],'height':img.shape[1], 'data':b64_img}
    response = requests.post('http://'+remote_address+':8080', data=b64_img)
    print(response.status_code)    # HTTPのステータスコード取得
    print(response.text)    # レスポンスのHTMLを文字列で取得

while(1):
    ret, frame = cap.read()
    # print(frame.shape)
    frame = cv2.resize(frame,(FRAME_W,FRAME_H))
    fgmask = fgbg.apply(frame) # マスク取得

    kernel = np.ones((3,3),np.uint8)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN,kernel)
    kernel = np.ones((10,10),np.uint8)
    # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE,kernel)
    masked_frame = cv2.bitwise_and(frame,frame,mask=fgmask)
    cv2.imshow('masked', masked_frame)

    Y, X = np.where(fgmask > 200)
    # print(Y.shape[0])
    if np.max(fgmask) > 200 and Y.shape[0] > 5000:
        #差分箇所の座標と範囲を取得
        Y, X = np.where(fgmask > 200)
        x = int(np.min(X)-margin)
        if x < 0: #マイナスならマージンなし
            x = x + margin
        w = int(np.max(X)-x+1+margin)
        # if x + w > frame.shape[1]: #マージンで幅の最大値超えたらマイナスする
        #     w = w-x+w-frame.shape[1]
        y = int(np.min(Y)-margin)
        if y < 0: #マイナスならマージンなし
            y = y + margin
        h = int(np.max(Y)-y+1+margin)
        # if y + h > frame.shape[0]:  #マージンで縦の最大値超えたらマイナスする
        #     h = h - y + h - frame.shape[0]
        #検出範囲が一定以下は画像を検出しない
        # print(w,h)
        if w > rangethreshold and h > rangethreshold:
            print("検出成功　　座標　x:{} y{} w {} h {}".format(x, y, w, h))
            # detected_image = frame[y:y + h, x:x + w] # 元画像から切り出し
            # detected_image = fgmask[y:y + h, x:x + w] # マスクから切り出し
            detected_image = masked_frame[y:y + h, x:x + w] # マスクされた画像から切り出し

            # print("detected_image取得")
            # zeropad = _zero_padding(frame, x, y, w, h)
            # print("zeropadding取得")
            # boundbox = deepcopy(frame)
            # boundbox = self._bounding_box(boundbox, x, y, w, h)
            # print("boundbox取得")
            cv2.imshow('detect', detected_image)
            detected_frame.append(detected_image)
            count += 1
            current_time += 1 
            if current_time > START_TIME:
                mode = 1
        else:
            if mode == 1:
                current_time = 0
                mode = 0
                frame_len = len(detected_frame)
                print(frame_len)
                send_image(detected_frame[frame_len//5+2]) # 検出した画像のうち実際に送信するフレームを選択
                detected_frame = []
            # pass
            #print("検出物体が小さい")

        # if count >= 5:
        #     cv2.imshow('queue',detected_frame[2])
    else:
        if mode == 1:
            current_time = 0
            mode = 0
            frame_len = len(detected_frame)
            print(frame_len)
            send_image(detected_frame[frame_len//5+2]) # 検出した画像のうち実際に送信するフレームを選択
            detected_frame = []

    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
