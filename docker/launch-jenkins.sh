#!/bin/bash

docker build -t finn_jenkins -f Dockerfile.jenkins .
docker run -p 8080:8080 -p 50000:50000 -v /var/run/docker.sock:/var/run/docker.sock -v jenkins_home:/var/jenkins_home finn_jenkins
