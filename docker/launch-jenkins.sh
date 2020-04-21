#!/bin/bash

# defaults, can be overriden by environment variables
# user to run Jenkins as -- see NOTE below regarding Docker access permissions
JENKINS_USER ?= jenkins
# port for Jenkins on host machine
JENKINS_PORT ?= 8080
# make Jenkins config persistent by mounting into this folder
JENKINS_HOME ?= $(pwd)/jenkins_home

mkdir -p $JENKINS_HOME

# build a Jenkins Docker image that also has the Docker CLI installed
docker build -t finn_jenkins -f Dockerfile.jenkins .

# launch Docker container mounted to local Docker socket
# NOTE: we allow customizing the user (e.g. as root) to work around permission
# issues, may not al
docker run -u $JENKINS_USER -p $JENKINS_PORT:8080 -v /var/run/docker.sock:/var/run/docker.sock -v $JENKINS_HOME:/var/jenkins_home finn_jenkins
