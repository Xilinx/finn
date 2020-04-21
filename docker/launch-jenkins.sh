#!/bin/bash

JENKINS_USER ?= jenkins
JENKINS_PORT ?= 8080
JENKINS_HOME ?= $(pwd)/jenkins_home

mkdir -p $JENKINS_HOME

docker build -t finn_jenkins -f Dockerfile.jenkins .
docker run -u $JENKINS_USER -p $JENKINS_PORT:8080 -v /var/run/docker.sock:/var/run/docker.sock -v $JENKINS_HOME:/var/jenkins_home finn_jenkins
