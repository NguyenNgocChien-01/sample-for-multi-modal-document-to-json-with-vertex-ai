#!/usr/bin/env bash
set -x
# This script builds a Docker image and pushes it to Artifact Registry for use with Vertex AI
# Arguments: repo-name, version-tag, region, project-id

reponame=$1
versiontag=$2
regionname=$3
project=$4

# Validate input parameters
if [ "$reponame" == "" ] || [ "$versiontag" == "" ] || [ "$regionname" == "" ] || [ "$project" == "" ]
then
   echo "Usage: $0 <repo-name> <version-tag> <region> <project-id>"
   exit 1
fi

fullname="${regionname}-docker.pkg.dev/${project}/${reponame}/inference:${versiontag}"

# Check if repository exists in Artifact Registry, create if it doesn't
echo "Checking Artifact Registry repository..."
gcloud artifacts repositories describe "${reponame}" --location="${regionname}" --project="${project}" > /dev/null 2>&1
if [ $? -ne 0 ]
then
   echo "Creating Artifact Registry repository ${reponame}..."
   gcloud artifacts repositories create "${reponame}" \
       --repository-format=docker \
       --location="${regionname}" \
       --project="${project}" \
       --quiet
   if [ $? -ne 0 ]; then
       echo "Error: Failed to create Artifact Registry repository"
       exit 1
   fi
fi

# Configure Docker to authenticate with Artifact Registry
echo "Configuring Docker auth for Artifact Registry..."
gcloud auth configure-docker "${regionname}-docker.pkg.dev" --quiet
if [ $? -ne 0 ]; then
   echo "Error: Docker auth configuration failed"
   exit 1
fi

# Build docker image
echo "Building docker image..."
pwd
docker build -f Dockerfile -t ${reponame} .
if [ $? -ne 0 ]; then
   echo "Error: Docker build failed"
   exit 1
fi

# Tag image
echo "Tagging docker image..."
docker tag ${reponame} ${fullname}
if [ $? -ne 0 ]; then
   echo "Error: Docker tag failed"
   exit 1
fi

# Push to Artifact Registry
echo "Pushing image to Artifact Registry..."
docker push ${fullname}
if [ $? -ne 0 ]; then
   echo "Error: Docker push failed"
   exit 1
fi

# Save image URI to file
echo "Saving image URI to file..."
echo "${fullname}" > vertex-image-uri.txt
if [ $? -ne 0 ]; then
   echo "Error: Failed to save image URI to file"
   exit 1
fi

echo "Successfully built and pushed image: ${fullname}"
echo "Image URI saved to: vertex-image-uri.txt"