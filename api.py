from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np
import base64
from rembg import remove
from model_manager import ModelManager
from schema import Config
import uvicorn
import cv2
from image_utils import read_text, get_mask, draw_text

config = Config(
    ldm_steps=1,
    hd_strategy="1",
    hd_strategy_crop_margin=2,
    hd_strategy_crop_trigger_size=1,
    hd_strategy_resize_limit=2
)

model = ModelManager(name='lama', device='cpu')


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def base64_image(base64_str):

    # Assuming base64_str is the string value without 'data:image/jpeg;base64,'
    img = Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))
    return np.asarray(img)

def image2base64(img):
    _, im_arr = cv2.imencode('.png', img)  # im_arr: image in Numpy one-dim array format.
    im_bytes = im_arr.tobytes()
    im_b64 = base64.b64encode(im_bytes)
    return "data:image/png;base64," + str(im_b64, 'utf-8')

@app.post("/inpaint")
async def inpaint(
    request: Request
    ):
    data = await request.json()
    image = base64_image(data['image'].split(",")[-1])
    mask_numpy = base64_image(data['mask'].split(",")[-1])
    img_height, img_width = image.shape[:2]
    mask_numpy = cv2.resize(mask_numpy, (img_width, img_height), cv2.INTER_AREA)
    mask_numpy = cv2.cvtColor(mask_numpy, cv2.COLOR_RGB2GRAY)
    mask = cv2.threshold(mask_numpy, 254, 255, cv2.THRESH_BINARY)[1]
    out_img = model(image = image, mask = mask, config=config)
    base64_img = image2base64(out_img)
    return {"image_base64": base64_img}


@app.post("/autoinpaint")
async def autoinpaint(
    request: Request
    ):
    data = await request.json()
    # image_byte = image.file.read()
    # image = np.asarray(Image.open(io.BytesIO(image_byte))).astype(np.uint8)
    image = base64_image(data['image'].split(",")[-1])
    detail, bboxes = read_text(image)
    mask = get_mask(image, detail)
    
    out_img = model(image, mask, config=config)
    base64_uri = image2base64(out_img)
    return {"image_base64": base64_uri, "result": detail}


@app.post("/autotranslation")
async def autoTrans(
    request: Request
    ):
    data = await request.json()
    image = base64_image(data['image'].split(",")[-1])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detail, bboxes = read_text(image)
    mask = get_mask(image, detail)
    out_img = model(image, mask, config=config)
    out_img = out_img.astype(np.uint8)
    for item in detail:
        out_img = draw_text(item, out_img)
    
    out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
    base64_uri = image2base64(out_img)
    return {"image_base64": base64_uri}
    
    
@app.post("/removebg")
async def removebg(
    request: Request
    ):
    data = await request.json()
    image = base64_image(data['image'].split(",")[-1])
    out_img = remove(image)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_BGRA2RGBA)
    base64_uri = image2base64(out_img)
    return {"image_base64": base64_uri}


if __name__=="__main__":
    uvicorn.run("api:app", workers=4)