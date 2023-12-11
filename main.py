from model import build_MSAPVT

import numpy as np
import gradio as gr
import torch
from torchvision import transforms


model = build_MSAPVT()
model_weight_pth = 'model_weight.pth'

model.load_state_dict(torch.load(model_weight_pth,map_location='cpu'))
model.eval()

labels = fruit_names = [
    'almond', 'annona_muricata', 'apple', 'apricot', 'artocarpus_heterophyllus',
    'avocado', 'banana', 'bayberry', 'bergamot_pear', 'black_currant', 'black_grape',
    'blood_orange', 'blueberry', 'breadfruit', 'candied_date', 'carambola', 'cashew_nut',
    'cherry', 'cherry_tomato', 'Chinese_chestnut', 'citrus', 'coconut', 'crown_pear',
    'Dangshan_Pear', 'dekopon', 'diospyros_lotus', 'durian', 'fig', 'flat_peach', 'gandaria',
    'ginseng_fruit', 'golden_melon', 'grape', 'grapefruit', 'grape_white', 'green_apple',
    'green_dates', 'guava', 'Hami_melon', 'hawthorn', 'hazelnut', 'hickory', 'honey_dew_melon',
    'housi_pear', 'juicy_peach', 'jujube', 'kiwi_fruit', 'kumquat', 'lemon', 'lime', 'litchi',
    'longan', 'loquat', 'macadamia', 'mandarin_orange', 'mango', 'mangosteen', 'munlberry',
    'muskmelon', 'naseberry', 'navel_orange', 'nectarine', 'netted_melon', 'olive', 'papaya',
    'passion_fruit', 'pecans', 'persimmon', 'pineapple', 'pistachio', 'pitaya', 'plum',
    'plum-leaf_crab', 'pomegranate', 'pomelo', 'ponkan', 'prune', 'rambutan', 'raspberry',
    'red_grape', 'salak', 'sand_pear', 'sugarcane', 'sugar_orange', 'sweetsop', 'syzygium_jambos',
    'trifoliate_orange', 'walnuts', 'wampee', 'wax_apple', 'winter_jujube', 'yacon'
]


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
    return image
    
def predict(img):
    data_transform = transforms.Compose([
       transforms.Resize((224, 224)),
       transforms.ToTensor(),
       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
       ])
    
    img = data_transform(cvtColor(img)).unsqueeze(0)
    with torch.no_grad():
      prediction = torch.nn.functional.softmax(model(img)[0], dim=0)
      confidences = {labels[i]: float(prediction[i]) for i in range(92)}
    return confidences


if __name__=='__main__':
    examples=["Pink-Lady_003.jpg","Avocado_007_2.jpg","Banana_029_2.jpg","Kiwi_002_2.jpg","Lime_006_2.jpg","Lemon_003_2.jpg"]
    gr.Interface(fn=predict,
                 inputs=gr.Image(type="pil", label="Fruit"),
                 outputs=gr.Label(num_top_classes=8, label="Predict confidences"),
                 live=True,
                 examples=examples).launch(share=True)