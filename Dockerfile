#FROM eidos-service.di.unito.it/eidos-base-pytorch:1.10.0
#FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
FROM eidos-service.di.unito.it/eidos-base-pytorch:1.12.0


RUN pip3 install filelock
RUN pip3 install gdown
RUN pip3 install einops
#TODO remover depois de extrair os multiviews
RUN pip3 install plenopticam
RUN pip3 install IPython




# Copy source files and make it owned by the group eidoslab
# and give write permission to the group
COPY src /src
RUN chmod 775 /src
RUN chown -R :1337 /src

# Do the same with the data folder
# RUN mkdir /data
# RUN chmod 775 /data
# RUN chown -R :1337 /data

WORKDIR /src

ENTRYPOINT ["python3"]