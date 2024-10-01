## Suggested pull command (run from anywhere):
## CUDA_VERSION=11.8 docker pull cbica/nichart:1.0.1-cuda${CUDA_VERSION}
## OR
## docker pull cbica/nichart:1.0.1

## Suggested automatic inference run time command 
## Place input in /path/to/input/on/host.
## Each "/path/to/.../on/host" is a placeholder, use your actual paths!
## docker run -it --name nichart_server --rm -p 8501:8501
##    --mount type=bind,source=/path/to/input/on/host,target=/input,readonly 
##    --mount type=bind,source=/path/to/output/on/host,target=/output
##    --gpus all cbica/nichart:1.0.1
## Run the above, then open your browser to http://localhost:8501 
## The above runs the server in your terminal, use Ctrl-C to end it.
## To run the server in the background, remove the "-it" flag in the command.
## To end the background server, use "docker stop nichart_server"
## DO NOT USE this as a public web server!

## Suggested build command (run from repo after pulling submodules):
## CUDA_VERSION=11.8 docker build --build-arg CUDA_VERSION=${CUDA_VERSION} 
##      -t cbica/nichart:1.0.1-cuda${CUDA_VERSION} .
## OR
## docker build -t cbica/nichart:1.0.1 .

ARG NICHART_VERSION="1.0.1"
ARG CUDA_VERSION="11.8"
ARG TORCH_VERSION="2.3.1"
ARG CUDNN_VERSION="8"

## This base image is generally the smallest with all prereqs.
FROM pytorch/pytorch:${TORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime

WORKDIR /app
COPY . /app/ 
RUN apt-get update && apt-get install build-essential -y
RUN pip install .
RUN mkdir /dummyinput && mkdir /dummyoutput
## Cache DLMUSE and DLICV models with an empty job so no download is needed later
RUN DLMUSE -i /dummyinput -o /dummyoutput && DLICV -i /dummyinput -o /dummyoutput
WORKDIR /app/src/NiChart_Viewer/src
ENTRYPOINT ["streamlit", "run", "NiChartProject.py", "--server.headless", "true"]