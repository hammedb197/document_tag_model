FROM python:3.8-slim-buster

RUN apt-get update -y

# gcc compiler and opencv prerequisites
RUN apt-get -y install nano git build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev

# Detectron2 prerequisites
RUN pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install cython
RUN pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# Detectron2 - CPU copy
RUN python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/index.html

# Development packages
RUN pip install flask flask-cors requests opencv-python

RUN apt -y install tesseract-ocr
RUN apt-get -y install libtesseract-dev
RUN apt-get -y install tesseract-ocr

RUN pip install pandas
RUN pip install numpy
RUN pip install pytesseract
RUN pip install imutils
RUN pip install textract

WORKDIR /app

COPY api.py api.py
COPY extracting.py extracting.py
COPY extract_from_image.py extract_from_image.py

ENTRYPOINT ["python", "/app/api.py"]