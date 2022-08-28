FROM python:3.8-slim

ENV PATH="/pricetag/:${PATH}"
ENV PORT=5005
ENV HOST="0.0.0.0"


COPY ./reqs.txt /reqs.txt

RUN apt update
RUN apt upgrade  --yes  
RUN apt install --yes  libzbar-dev libzbar0
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r reqs.txt
# RUN pip3 install numpy --upgrade


RUN mkdir /pricetag
RUN mkdir /pricetag/api
RUN mkdir /pricetag/engine
RUN mkdir /pricetag/pipeline
RUN mkdir /pricetag/utils
RUN mkdir /media/models
RUN mkdir /media/models/production


COPY ./api /pricetag/api
COPY ./engine /pricetag/engine
COPY ./pipeline /pricetag/pipeline
COPY ./utils /pricetag/utils
COPY ./config.py /pricetag/config.py
COPY ./logging_cfg.yaml /pricetag/logging_cfg.yaml
COPY ./main.py /pricetag/main.py
COPY ./tf_load.py /pricetag/tf_load.py
COPY ./engine_config.json /pricetag/
COPY ./entrypoint.sh /pricetag/

WORKDIR /pricetag

CMD python3 main.py --host ${HOST} --port ${PORT} --config engine_config.json
# ENTRYPOINT ["/pricetag/entrypoint.sh"]
EXPOSE ${PORT}
