import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from starlette.responses import HTMLResponse
from tensorflow.keras.models import model_from_json
from tensorflow.keras.backend import clear_session
from PIL import Image
import numpy as np
import io
import sys

app = FastAPI()


def load_model():
    json_file = open("model/model.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("model/model.h5")
    return model


@app.get('/upload', response_class=HTMLResponse)
def upload_file():
    return '''
        <form action="/predict" method="post" enctype="multipart/form-data">
            Select image to upload:
            <input type="file" name="file">
            <input type="submit">
        </form>
        '''


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    if file.content_type.startswith('image/') is False:
        raise HTTPException(status_code=400, detail=f'File \'{file.filename}\' is not an image.')

    try:
        # Read input image
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents))

        # Resize input image to expected shape
        input_shape = (None, 224, 224, 3)

        pil_image = pil_image.resize((input_shape[1], input_shape[2]))

        # Convert RGBA to RGB
        if pil_image.mode == 'RGBA':
            pil_image = pil_image.convert('RGB')

        # Convert image into numpy format
        numpy_image = np.array(pil_image).reshape((input_shape[1], input_shape[2], input_shape[3]))

        # Scale data (depending on your model)
        numpy_image = numpy_image / 255

        images_list = []
        images_list.append(np.array(numpy_image))
        x = np.asarray(images_list)

        clear_session()
        model = load_model()

        prediction = model.predict(x, batch_size=len(x)).tolist()[0][0]
        image_class = round(prediction)
        probability = "{:.0%} skin cancer".format(prediction)

        return {
            'predicted_class': image_class,
            'details': probability
        }

    except:
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
