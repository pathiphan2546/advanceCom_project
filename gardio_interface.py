#ทำใน Google colab
import gradio as gr
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import pygame
import requests

pygame.mixer.init()

notification_sound = pygame.mixer.Sound('door-bang-1wav-14449.mp3')
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
LINE_NOTIFY_TOKEN = 'Qnkyfpincr09QcOvtZ1qKuKNSWOFaY1Mffk7AL9IoQQ'

def send_line_notification(message):
    print(f'Sending LINE notification: {message}')
    url = 'https://notify-api.line.me/api/notify'
    headers = {
        'Authorization': f'Bearer {LINE_NOTIFY_TOKEN}'
    }
    data = {
        'message': message
    }
    response = requests.post(url, headers=headers, data=data)
    if response.status_code != 200:
        print('Failed to send LINE notification')

def detect_potholes(input_image, confidence_threshold=0.05):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image).unsqueeze(0)

    with torch.no_grad():
        prediction = model(input_tensor)

    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']

    pothole_detected = False

    for label, score in zip(labels, scores):
        if label == 1 and score >= confidence_threshold:
            pothole_detected = True
            break

    if pothole_detected:
        return "No Pothole Detected"
    else:
        notification_sound.play()
        send_line_notification("Pothole Detected")
        return "Pothole Detected"

input_image = gr.inputs.Image(shape=(1000, 1000))
iface = gr.Interface(
    fn=detect_potholes,
    inputs=input_image,
    outputs="text",
    live=True,
    title="Gardio Pothole Detection",
    description="Upload an image to check for potholes .",
    theme="default",
)
iface.launch()
