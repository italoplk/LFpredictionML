FROM eidos-service.di.unito.it/eidos-base-pytorch:1.10.0
RUN pip3 install filelock
RUN pip3 install gdown

# Copy source files and make it owned by the group eidoslab
# and give write permission to the group
COPY ./ ./
RUN chmod 775 ./
RUN chown -R :1337 ./

# Do the same with the data folder
# RUN mkdir /data
# RUN chmod 775 /data
# RUN chown -R :1337 /data

WORKDIR ./

ENTRYPOINT ["python3"]