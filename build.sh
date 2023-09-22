#!/bin/sh

docker build -t eidos-service.di.unito.it/machado/machado-test-image:latest . -f Dockerfile
docker push eidos-service.di.unito.it/machado/machado-test-image:latest