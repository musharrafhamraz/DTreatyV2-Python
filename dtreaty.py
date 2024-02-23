from kivy.core.text import LabelBase
from kivy.app import App
from kivy.properties import StringProperty, ObjectProperty
from kivy.uix.screenmanager import ScreenManager, Screen
from kivymd.app import MDApp
from kivy.uix.camera import Camera
from kivy.lang import Builder
from keras.preprocessing.image import load_img, img_to_array
from kivy.core.window import Window
import pandas as pd
from PIL import Image as PILImage
import tensorflow as tf
import tempfile
import numpy as np
import os

Window.size = (310, 580)

class CameraScreen(Screen):
    pass

class TreatmentScreen(Screen):
    pass

class Dtreaty(MDApp):
    condition_value = StringProperty()  # Define a StringProperty
    treatment = StringProperty()
    def build(self):
        screenmanager = ScreenManager()
        screenmanager.add_widget(Builder.load_file("main.kv"))
        screenmanager.add_widget(Builder.load_file("camera.kv"))
        screenmanager.add_widget(Builder.load_file("treatment.kv"))
        return screenmanager
    def capture_image(self):
        current_screen = self.root.current_screen
        if current_screen.name == 'camera':
            camera_widget = current_screen.ids.camera_id
            if camera_widget.texture:
                img_texture = camera_widget.texture
                if img_texture:
                    pil_image = PILImage.frombytes('RGBA', img_texture.size, img_texture.pixels)
                    pil_image = pil_image.convert('RGB')
                    pil_image = pil_image.resize((224, 224))  # Resize the image to the desired input size
                    image_array = img_to_array(pil_image)
                    image_array = image_array / 255.0
                    self.model = tf.keras.models.load_model('DTreatyVGG19.h5')
                    prediction = self.model.predict(np.expand_dims(image_array, axis=0))
                    predicted_class = np.argmax(prediction, axis=1)
                    class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
                    predicted_class_name = class_names[predicted_class[0]]
                    self.label = f"{predicted_class_name}"

    def read_data(self):
        df = pd.read_csv('treatment-book.csv', encoding='ISO-8859-1')
        condition_column = 'disease_name' 
        condition_value = self.label
        value_column = 'treatment'
        self.no_data = "No matching data found"

        self.condition_value = condition_value

        self.treatment_series = df.loc[df[str(condition_column)] == str(condition_value), str(value_column)]
        self.treatment_series = self.treatment_series.astype(str)
        treatment = "\n".join(self.treatment_series) if not self.treatment_series.empty else self.no_data

        self.treatment = treatment

if __name__ == "__main__":
    LabelBase.register(name="Poppins", fn_regular="D:\kivy\DTreatyApp\Poppins-Bold.ttf")
    LabelBase.register(name="Poppinsl", fn_regular="D:\kivy\DTreatyApp\Poppins-Light.ttf")

    Dtreaty().run()

