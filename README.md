# 

<div align="center">
   <h1>python_opencv_ocr_font_color</h1>
</div>

This code implements an automated image text translation system with OCR detection and text replacement capabilities

This program uses Google OCR to obtain text data from an image file, and uses OpenCV to obtain font size and color information from the text data.



<div align="center">
   <img src=https://github.com/LucaIT523/python_opencv_ocr_font_color/blob/main/images/1.png>
</div>



### 1. Core Architecture

**A. Main Pipeline**

```
Image Input → OCR Detection → Text Translation → Text Removal → Translated Text Rendering → Output Image
```

**B. Key Components**

1. Model Management Module
   - Model downloading & loading utilities
   - Device optimization (CPU/MPS)
2. Image Processing Core
   - Format conversion (numpy/PIL/cv2)
   - Adaptive padding/resizing
   - Mask generation
3. OCR & Translation Engine
   - PaddleOCR text detection
   - Naver Papago translation API
4. Text Rendering System
   - Font size calculation
   - Color analysis
   - Text positioning

### 2. Technical Implementation

**A. Model Operations** (`download_model`, `load_jit_model`)

```
pythonCopydef download_model(url):
    cached_file = get_cache_path_by_url(url)
    if not os.path.exists(cached_file):
        download_url_to_file(url, cached_file, ...)
```

- Implements model caching system
- Supports TorchScript models
- Automatic device detection (MPS/CPU)

**B. Image Preprocessing** (`pad_img_to_modulo`)

```
def pad_img_to_modulo(img, mod):
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(img, ...)
```

- Ensures image dimensions meet model requirements
- Uses symmetric padding for seamless processing

**C. OCR Processing** (`read_text`)

```
result = ocr.ocr(image, cls=False)
for detection in result[0]:
    top_left = tuple(int(val) for val in detection[0][0])
    ...
```

- Leverages PaddleOCR for multilingual detection
- Returns bounding boxes with text coordinates

**D. Text Analysis** (`get_color`, `find_font_size`)

```
pythonCopydef get_color(...):
    colors = sorted(image_roi.getcolors(...))
    return colors[-2][-1]  # Second dominant color
```

- Color analysis for adaptive text rendering
- Automatic font sizing based on bounding box dimensions

**E. Translation Service** (`translate`)

```
data = f"source={source}&target=ko&text={encText1}"
request = urllib.request.Request(url)
...
return json.loads(response_body)['translatedText']
```

- Automatic language detection (Korean target)
- Fallback to original text on API failure

### 3. Key Features

**A. Text Replacement Workflow**

1. Generate text region masks
2. Remove original text via inpainting
3. Render translated text with:
   - Context-aware font sizing
   - Background contrast analysis
   - Position alignment

**B. Performance Optimization**

- Batch text box processing
- Local model caching
- Asynchronous translation API calls

**C. Image Quality Preservation**

- Lossless format conversion
- Adaptive thresholding
- High-quality upscaling (INTER_CUBIC)

### 4. Usage Scenarios

1. Document Localization
   - Technical manuals
   - Educational materials
2. Marketing Assets Adaptation
   - Advertisement graphics
   - Product packaging
3. Mobile App Content Generation
   - Game asset localization
   - In-app screenshots





### **Contact Us**

For any inquiries or questions, please contact us.

telegram : @topdev1012

email :  skymorning523@gmail.com

Teams :  https://teams.live.com/l/invite/FEA2FDDFSy11sfuegI