import PySimpleGUI as sg
from PIL import Image, ImageTk
from uio import runner
import numpy as np
from threading import Thread

layout = [
    [sg.Text("Question:"), sg.InputText(font=("Helvetica", 20))],
    [sg.Text("Image:"), sg.InputText(key="-FILE-", font=("Helvetica", 20)), sg.FileBrowse(font=("Helvetica", 20))],
    [sg.Submit(font=("Helvetica", 20)), sg.Cancel(font=("Helvetica", 20))],
    [sg.Image(key="-IMAGE-", size=(400, 400))],
    [sg.Text(size=(50, 1), key="-RESPONSE-", font=("Helvetica", 20))]
]

window = sg.Window("Common Sense?", layout, font=("Helvetica", 20))

model = runner.ModelRunner('small', 'small_1000k.bin')


def infer(img, question, window: sg.Window):
    image = np.array(img.convert('RGB'))
    output = model.vqa(image, question=question)
    window.write_event_value('infer', {'question': question, 'answer': output['text']})


while True:
    event, values = window.read()
    if event in (sg.WIN_CLOSED, "Cancel"):
        break
    if event == 'infer':
        print("done")
        window["-RESPONSE-"].update(f"Response to '{values['infer']['question']}': {values['infer']['answer']}")
    if event == "Submit":
        question = values[0]
        image_path = values["-FILE-"]

        img = Image.open(image_path)
        img = img.resize((400, 400))

        window["-IMAGE-"].update(data=ImageTk.PhotoImage(img))
        # Do something with the question and image_path
        print(f"Question: {question}\nImage Path: {image_path}")

        window["-RESPONSE-"].update(f"Response to '{question}': Pending ...")
        Thread(target=infer, args=(img, question, window)).start()
        # Update the image and response text


window.close()
