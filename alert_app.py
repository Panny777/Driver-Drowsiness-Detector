from playsound import playsound
import time

def alert():
    while True:
        playsound('sounds/alert.mp3')
        time.sleep(3)
        sys.exit()
alert()
