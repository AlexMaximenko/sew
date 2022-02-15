FROM cr.msk.sbercloud.ru/aicloud-base-images/cuda10.2:0.0.27

WORKDIR /tmp

RUN git clone https://github.com/pytorch/fairseq
RUN git clone https://github.com/microsoft/DeBERTa
RUN git clone https://github.com/AlexMaximenko/sew

USER root
#RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 --yes
RUN apt-get install libsndfile1 --yes

#Installing fairseq
RUN cd fairseq && git checkout 05255f96 && pip install -e .

#Installing deberta
RUN cd DeBERTa && git checkout bf17ca43fa429875c823536b5993cdf783ae5049 && pip install -e .

RUN cd sew && pip install --no-cache-dir -r requirements.txt

RUN pip install --upgrade importlib_metadata

WORKDIR /home/jovyan
