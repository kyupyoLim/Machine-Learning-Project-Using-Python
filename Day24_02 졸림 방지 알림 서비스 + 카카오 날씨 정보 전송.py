import tensorflow.keras
import numpy as np
import random
import cv2
import winsound as sd
import os
import time
import requests
import json
from bs4 import BeautifulSoup
import requests

#함수부
def beepsound():
    fr = 500    # range : 37 ~ 32767
    du = 1000     # 1000 ms ==1second
    sd.Beep(fr, du) # winsound.Beep(frequency, duration)
def beepsound2():
    fr = 2500    # range : 37 ~ 32767
    du = 2000     # 1000 ms ==1second
    sd.Beep(fr, du) # winsound.Beep(frequency, duration)
def weatherInfo():
    global address, temperature, weatherStatus, air, refreshmessage

    html = requests.get('https://search.naver.com/search.naver?query=날씨')
    soup = BeautifulSoup(html.text,'html.parser')

    #위치
    address = soup.find('div',{'class':'title_area _area_panel'}).find('h2',{'class':'title'}).text
    print(address)

    #날씨 정보
    weather_data = soup.find('div',{'class' : 'weather_info'})

    #현재 온도
    temperature = weather_data.find('div',{'class' : 'temperature_text'}).text.strip()[0:]
    print(temperature)

    #날씨 상태
    weatherStatus = weather_data.find('span',{'class' : 'weather before_slash'}).text
    print(weatherStatus)

    # 공기 상태
    air = soup.find('ul', {'class': 'today_chart_list'}).text.strip()
    print(air[0:7])
    print(air[12:20])
    print(air[25:31])
    print(air[36:44])
    # infos = air.find_all('li',{'class' : 'item_today'})
    #
    # for i in infos:
    #     print(i.text.strip())

    if ((air[0:7] == "미세먼지 보통") and (air[12:20] == "초미세먼지 보통")):
        refreshmessage = "미세먼지와 초미세먼지가 '보통'이므로 환기하시는게 좋을 듯 합니다."
        print(refreshmessage)
    elif ((air[0:7] == "미세먼지 나쁨") and (air[12:20] == "초미세먼지 나쁨")):
        refreshmessage = "미세먼지와 초미세먼지가 '나쁨'이므로 환기는 안하는 게 좋을 듯 합니다."
        print(refreshmessage)
    elif ((air[0:7] == "미세먼지 좋음") and (air[12:20] == "초미세먼지 좋음")):
        refreshmessage = "미세먼지와 초미세먼지가 '좋음'이므로 환기를 하는게 좋을 듯 합니다."
        print(refreshmessage)

def KaKaoMessage():
    global address, temperature, weatherStatus, air, refreshmessage
    # url = "https://kauth.kakao.com/oauth/token"
    # data = {
    #     "grant_type" : "authorization_code",
    #     "client_id" : "{REST API KEY}",
    #     "redirect_url" : "https://localhost.com",
    #     "code" : "{REPLAY CODE}"
    # }
    #
    # response = requests.post(url, data=data)
    # tokens = response.json()
    # print(tokens)
    #
    # with open("kakao_code.json", "w") as fp:
    #     json.dump(tokens, fp)
    #     print("OK")

    #날씨 호출
    weatherInfo()

    with open("kakao_code.json", "r") as fp:
        ts = json.load(fp)
        #print(ts)
        print("json 파일 읽기 ok")

    url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"

    headers = {
        "Authorization": "Bearer " + ts["access_token"]
    }
    template = {
        "object_type": "text",
            "text": "졸지말고 환기합시다잇!!!.\n__________________________________\n" + "\n" + address + "\n" + temperature + "\n" + weatherStatus + "\n" +
                    air[0:7] + "\n" + air[12:20] + "\n" + air[25:31] + "\n" + air[36:44] + "\n__________________________________\n\n" + refreshmessage,
        "link": {
            "web_url": "https://www.naver.com",
            "mobile_web_url": "https://www.naver.com"
        },
        "button_title": "확인"
    }
    data = {
        "template_object": json.dumps(template)
    }
    response = requests.post(url, headers=headers, data=data)
    if response.json().get('result_code') == 0:
        print('메시지를 성공적으로 보냈습니다.')
    else:
        print('메시지를 성공적으로 보내지 못했습니다. 오류메시지 : ' + str(response.json()))


## 전역 변수
address, temperature, weatherStatus, air = None, None, None, None
refreshmessage = None
model = tensorflow.keras.models.load_model('keras_model.h5')
capture = cv2.VideoCapture(0)

CLASSES = ['Wake up!','Good :)']
sleepCount = 0
while True :
    ret, frame = capture.read()

    if not ret:
        break

    frame = cv2.flip(frame,1)

    h,w,c = frame.shape
    frame = frame[:, 100:100 + h]  # 정사각형으로 자르기

    frame_input = cv2.resize(frame, (224, 224))  # image를 (224,224)로 변경하기

    ##이미지 전처리
    frame_input = cv2.cvtColor(frame_input, cv2.COLOR_BGR2RGB)  # BGR -> RGB

    frame_input = (frame_input.astype(np.float32) / 127.0) - 1.0
    frame_input = np.expand_dims(frame_input, axis=0) #(1,224,224,3)

    prediction = model.predict(frame_input)

    idx = np.argmax(prediction)
    cv2.putText(frame, text=CLASSES[idx], org=(10, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2,
                color=(255, 255, 255), thickness=2)

    if prediction[0, 0] > prediction[0, 1]:
        sleepCount += 1
        beepsound()
        print(sleepCount, '초 동안 졸고 있습니다 :(')

        if ((sleepCount % 5) == 0):
            sleepCount = 0
            beepsound2()
            print("이제 그만 스트레칭을 합시다")
            time.sleep(1)
            os.system('explorer https://youtu.be/mUnSpfItRf0')
            KaKaoMessage()
            break
    else:
        print('좋습니다.')
        sleepCount = 0
    cv2.imshow('No Sleep!', frame)
    key = cv2.waitKey(20)
    if key == 27:  # esc 키 누를면 종료
        break
    elif key == ord('s') or key == ord('S'):  # 키보드 C 누르면
        cv2.imwrite('save' + str(random.randint(1111111, 9999999)) + '.png', frame)
capture.release()
cv2.destroyAllWindows()