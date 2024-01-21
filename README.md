# ML-app-for-Classification
The project involves creating a Gradio app for butterfly classification. It utilizes a pre-trained butterfly classification model (model.h5) and class labels stored in a CSV file (classLabel1.csv). Users can upload images of butterflies to the app, and it provides real-time predictions for the butterfly's class label.Users can upload images of butterflies to the app, and it provides real-time predictions for the butterfly's class label. The code integrates Gradio, TensorFlow, and pandas to create an interactive and user-friendly interface for butterfly classification.

The source code combines the Gradio library, TensorFlow, and pandas to create a user-friendly interface for butterfly classification. The provided Gradio app enables users to interact with the pre-trained model and obtain predictions with ease.

## Here's a step-by-step description of how the code sets up a Gradio app for butterfly classification:

1. **Importing Libraries:**
   - The code starts by importing the necessary libraries:
     - `gradio`: Used for creating user interfaces for machine learning models.
     - `tensorflow`: Used for loading the pre-trained model.
     - `pandas`: Used for reading class labels from a CSV file.
     - `numpy`: Used for numerical operations.

```python
import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import pandas as pd
import numpy as np
```

2. **Loading the Trained Model:**
   - The code loads the pre-trained butterfly classification model (`model.h5`) using TensorFlow's `load_model` function.

```python
model_path = '/content/model.h5'
model = load_model(model_path)
```

3. **Loading Class Labels from CSV:**
   - The code reads class labels from a CSV file (`classLabel1.csv`) using the `pd.read_csv` function and converts them to a Python list.

```python
csv_file_path = '/content/classLabel1.csv'
df_class_labels = pd.read_csv(csv_file_path)
class_labels = df_class_labels['ClassLabel'].tolist()
```

4. **Defining the Prediction Function:**
   - The `predict_butterfly` function is defined to take an image as input, preprocess it, make predictions using the loaded model, and return the predicted class label.

```python
def predict_butterfly(img):
    # Preprocess the image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make predictions using the loaded model
    predictions = model.predict(img_array)

    # Custom decoding based on your model's classes
    top_prediction_index = np.argmax(predictions)
    top_prediction = class_labels[top_prediction_index]

    return top_prediction
```

5. **Creating Gradio Interface:**
   - The `gr.Interface` class is used to create the Gradio interface:
     - `fn`: Specifies the prediction function.
     - `inputs`: Set to `gr.Image()` to accept an image as input.
     - `outputs`: Set to `gr.Textbox()` to display the predicted class label as text.
     - `live`: Set to `True` for live updates.
     - `title`: Sets the title of the Gradio interface.
     - `description`: Provides a description of the interface.

```python
iface = gr.Interface(
    fn=predict_butterfly,
    inputs=gr.Image(),
    outputs=gr.Textbox(),
    live=True,
    title='Butterfly Classification App',
    description='Upload an image of a butterfly to get the predicted class label.'
)
```

6. **Launching the Gradio App:**
   - The `iface.launch()` method is called to launch the Gradio app, providing a link to the app.

```python
iface.launch()
```

7. **Running the Code:**
   - After running the code, users can click on the provided link to open the Gradio app in a new tab.
   - The app allows users to upload an image of a butterfly, and it provides real-time predictions for the butterfly's class label.

This description outlines how the code combines the Gradio library, TensorFlow, and pandas to create a user-friendly interface for butterfly classification. The provided Gradio app enables users to interact with the pre-trained model and obtain predictions with ease.
