import requests

resp = requests.post("https://flask-pytorch-mnist.herokuapp.com/", files={'file': open("three.png", 'rb')})

print(resp.text)