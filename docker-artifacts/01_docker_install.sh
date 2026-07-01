#!/bin/bash

sudo apt-get -y update
sudo apt-get -y install ca-certificates curl gnupg

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor --yes -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get -y update

# Auto-detect the latest stable docker-ce-cli version for this Ubuntu/Debian codename
# This ensures compatibility with the actual OS instead of hardcoding a specific version
echo "Detecting available Docker CE versions..."
VERSION_STRING=$(apt-cache madison docker-ce-cli | awk '{ print $3 }' | head -n1)

if [ -z "$VERSION_STRING" ]; then
    echo "Error: Could not determine Docker CE version to install"
    exit 1
fi

echo "Installing Docker CE CLI version: $VERSION_STRING"
sudo apt-get install docker-ce-cli=$VERSION_STRING docker-compose-plugin -y

sudo apt-get install -y docker-compose

# Validate the Docker Client is able to access Docker Server at [unix:///docker/proxy.sock]
echo "Validating Docker installation..."
docker version
docker-compose --version