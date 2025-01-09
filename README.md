# Chicken Scratch Recognition

1. [Introduction](#chicken-scratch-recognition)
2. [Streamlit - FastAPI - PyTorch & Hugging Face](#streamlit---fastapi---pytorch--hugging-face)
    - [Streamlit](#streamlit)
    - [FastAPI](#fastapi)
    - [PyTorch & Hugging Face](#pytorch--hugging-face)
3. [The good, bad, and ugly software development practices](#the-good-bad-and-ugly-software-development-practices)
4. [Running the services locally](#running-the-services-locally)
5. [Running on Docker](#running-on-docker)
    - [Docker Compose](#docker-compose)
    - [Tuning the Images](#tuning-the-images)
6. [Google Cloud Run Deployment](#google-cloud-run-deployment)
    - [Install Google Cloud CLI](#install-google-cloud-cli)
    - [Build and Push Docker Images to Google Container Registry (GCR)](#build-and-push-docker-images-to-google-container-registry-gcr)
    - [Deploy the Containers to Google Cloud Run](#deploy-the-containers-to-google-cloud-run)
    - [Test the Deployed Service](#test-the-deployed-service)
    - [Restrict Public Access to the Backend](#restrict-public-access-to-the-backend)
7. [Debugging and Fixing Some Issues](#debugging-and-fixing-some-issues)
8. [Suspend the Services (Scale to Zero) and Resume a Service (Scale Back Up)](#suspend-the-services-scale-to-zero-and-resume-a-service-scale-back-up)
9. [GitHub Google Cloud Deployment Pipeline](#github-google-cloud-deployment-pipeline)
10. [Fine-Tuning the Inference Time](#fine-tuning-the-inference-time)
     - [Optimize the PyTorch Model](#optimize-the-pytorch-model)
     - [Docker Image Size Optimization](#docker-image-size-optimization)
     - [TorchScript](#torchscript)
     - [GPU Support](#gpu-support)


## Introduction

The goal of this project was to learn how to build a machine learning (ML) application, containerize it with Docker, and deploy it to the cloud. The focus is more on the deployment than the software development or machine learning. This documentation serves as a reminder and notes what, why, and how I did it for future use. The "stack" used is Streamlit, FastAPI, PyTorch, Hugging Face, Docker, and Google Cloud Run.

The internet and YouTube are full of examples of how to make a "single page" ML app with Streamlit, run it in a Docker container, and deploy it to the cloud. However, there aren't that many examples of how separate front-end and back-end containers can be deployed to the cloud.

The idea for the application is not my own. I encountered a [GitHub repo](https://github.com/mafda/ml_with_fastapi_and_streamlit/) when I googled how to run a Streamlit and FastAPI ML app in Docker. At first, I had some difficulties following how the application worked. I also discovered that the Docker containers were quite big (3.31GB & 3.68GB), and it lacked cloud deployment instructions, so I decided to make my own version of it from scratch and document how I did it.

In the app, the user can draw a number, or more like a chicken scratch, on the Streamlit canvas. When the user presses the "Classify" button, the image on the canvas is sent to the FastAPI endpoint, where it gets processed by PyTorch using a pretrained neural network model from Hugging Face. The model returns probabilities for the number, or Chicken Scratch, and prints the most probable number as a prediction.

## Streamlit - FastAPI - PyTorch & Hugging Face

Here are some main challenges I encountered with the applications. Issues with Docker and Google Cloud Run are covered in their own sections.

### Streamlit

[Streamlit](https://streamlit.io/) is easy to install and use. The [Drawable Canvas](https://pypi.org/project/streamlit-drawable-canvas/) is a custom component. The most "challenging" parts with Streamlit are how to get the user experience tolerable. For example, how to get the text fields updated automatically after some button is pressed and data is retrieved from the backend server. The ```session_state``` variable and ```rerun()``` function turned out to be useful.

### FastAPI

[FastAPI](https://fastapi.tiangolo.com/) is also easy to set up. Basically, the only notable change that caused some issues later was to use the "classify" folder instead of the root to serve as an endpoint. Issues came because I forgot to add the folder to the URL. I also used BaseModel and field_validator from pydantic, and List from typing to validate the requests for the backend. Validation would have been handy for testing, but because I didn't write any tests, the benefits were more cosmetic.

### PyTorch & Hugging Face

The ML [model](https://huggingface.co/farleyknight/mnist-digit-classification-2022-09-04) used for the classification is a fine-tuned version of [google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k) on the MNIST dataset. The model performs reasonably well in this use case scenario, but the size of it (343 MB) caused some thinking when creating Docker images, and especially when deploying the images on the cloud. I tested some other smaller models, but the results were bad. The problem with the smaller models was that they were different architectures trained with the MNIST dataset. MNIST images are sized 28x28, while the Streamlit app canvas size is 280x280. Probably downscaling images to 28x28 caused noise and drift, and finding optimal normalization factors would have required some work. The fine-tuned model was trained with an image size of 224x224, which made scaling easier.

And like always with neural networks, transforming the image input data to a suitable PyTorch tensor for the model required some wondering. Finally, it took a while to discover where Hugging Face stores the models. My conclusion was that it stores them in some cache at runtime, and if you want to load it later on instead of pulling it from the repository, you need to first explicitly store it in a specific location and load it from there.

The used model performs well in classification, and the execution time is not an issue in local use. However, in a cloud environment, the size of the model and its execution time will become an issue. The model could and should be revisited, and I have listed some options in the [Optimize the PyTorch Model](#optimize-the-pytorch-model) section of this README.

### The good, bad, and ugly software development practices

From time to time, I have followed some good software development practices, and from time to time, I haven't. So some of the code is good, some is bad, and sometimes it might be ugly.

To mention some glimpses of good practices: use of virtual environment (conda), git, functions and classes, error handling (try/except), separation of concerns, dependency injection, environment variables, avoiding tight coupling, optimizing performance, use of Python libraries and frameworks, containerization, and documentation.

To the bad and ugly section goes not using PEP 8, minimal error handling, no tests, no logging, no CI/CD (this is on the to-do list though), no consistent dependency management (importing own classes and functions is sometimes difficult), running processes as a root, lack of documentation or too much documentation which makes the code harder to read.

## Running the services locally

I have used conda with Python version 3.10.16, but venv works as well. Create the conda environment and activate it:
```
conda create --name env_name python=3.10
conda activate env_name
```

Pip is not necessarily the best option to install packages in conda, but let's use it anyway:
```
pip install -r requirements.txt
```
The directory structure of the project is as follows:
```
Chicken-Scratch-Recognition/
├── backend
│   ├── Dockerfile
│   ├── main.py
│   ├── model.py
│   ├── models
│   │   ├── config.json
│   │   └── model.safetensors
│   ├── requirements.txt
│   └── schemas.py
├── docker-compose.backend_network.yml
├── docker-compose.private.yml
├── docker-compose.yml
├── frontend
│   ├── app.py
│   ├── backend_api.py
│   ├── Dockerfile
│   └── requirements.txt
├── README.md
├── requirements.txt
└── run.sh
```

Go to the frontend directory and start Streamlit:

```
streamlit run app.py
```

Then go to localhost:8501

To start FastAPI, go to the backend directory and run:

```
uvicorn backend.main:app --reload
```

Browse to ```http://127.0.0.1:8000/docs``` to verify that FastAPI is up and running.

It is a bit tedious to start the applications from different directories. You can start the applications from the project root directory by running:

```
streamlit run frontend/app.py
PYTHONPATH=backend uvicorn backend.main:app --reload
```

The FastAPI application imports the ImageClass, the PyTorch model, and the preprocessing and inference functions from the files which are in the same directory. However, because these imports (paths) are relative to the current working directory, the imports fail if the application is started from the project root directory. We can solve this by setting the PYTHONPATH environment variable to the folder where to look for the packages when starting the application.

Using ```PYTHONPATH``` is a bit of an ugly solution, and a more elegant way would be to use a ```.env``` file and Python ```dot-env``` with ```os.path``` libraries to set environment variables and build absolute or relative paths. I have used os.path for reference in the ```model.py``` file so that the PyTorch model can be saved and loaded locally instead of loading it from the Hugging Face repository. But in general, how the paths and imports are handled in the code is more ugly than good. Maybe I will fix this at some point, maybe not.

There can be, and in this case, there will be situations where you want to specify in which address and port the servers run:

```
streamlit run frontend/app.py --server.address 0.0.0.0 --server.port 8051
uvicorn backend.main:app --host 0.0.0.0 --port 8080 --reload
```

Be cautious if you run the servers locally (not inside the container) on ```0.0.0.0```, because you expose them basically to the whole world. This is not that big of an issue because your home network is probably behind a router's NAT, but it is good to keep in mind. This might sound confusing because with Docker, we need to run the servers on ```0.0.0.0``` to get things to work.

## Running on Docker

At this point, in both applications' Dockerfiles, the most important points to note are the commands used to run the containers and the server address where they run inside the container.

First, I have used CMD to start the servers with a shell command `sh -c`. This is because it makes it easier to pass arguments when running the containers. For example, the port where the server will run can be passed as an environment variable. And if it is not passed, the default value is used. Using a shell command is much more convenient in this case than using `CMD` and `ENTRYPOINT`.

Second, the servers are run on `0.0.0.0`. But as a distinction to the earlier, it is not the __host machine's__ network address, it is the Docker container's private network address __inside__ the container. Using `0.0.0.0` means that the server is listening (binding a service) on all network interfaces of the __container__. The container will accept connections from any IP address, both from the host machine and other containers in the network.

When exposing the container port and mapping it to the host's port, we can access the server using `0.0.0.0:port`, `127.0.0.1:port`, or `localhost:port` addresses from our host machine. However, if the server is set to run on `127.0.0.1` or `localhost` (loopback interface) instead of `0.0.0.0`, it cannot be accessed from the host even though the servers are running. This can cause confusion because the server is running on `127.0.0.1:port`, but it cannot be accessed, for example, from a browser by going to `127.0.0.1:port`. The reason is that if the server inside the container is bound to `127.0.0.1`, it will only be available within the container itself because the address refers to the container's loopback interface, not the host's. It listens only to the container's internal network and doesn't expose the port on the Docker bridge or host's network. Using `-p <host_port>:<container_port>` does not work.

To run the applications in Docker:

1. Build the images (in the respective folders)

```
docker build -t streamlit-app .
docker build -t fastapi-app .
```

2. Run the containers (in the respective folders)

```
docker run -d -p 8501:8501 --name frontend streamlit-app
docker run -d -p 8080:8080 --name backend fastapi-app
```

Go to `0.0.0.0:8051`, `127.0.0.1:8051`, or `localhost:8051` and you should land on the Streamlit page. And to access the backend (FastAPI), just change the port to and concatenate the endpoint `8080/docs`.

The servers are now up and running, but you'll encounter an issue when you press the "Classify" button. The Streamlit front-end tries to send the image for classification to the FastAPI server's "/classify" endpoint, but the request is refused. The issue arises, as described earlier, because the Streamlit and FastAPI servers run on different networks despite using the same network address but different ports. Though the servers use the same `0.0.0.0` address, they have their own distinct networks inside the containers! The Streamlit app does not know how to route the data to the FastAPI endpoint.

One solution is to assign the containers to the same Docker network. If the containers are on the same bridge network, they can communicate with each other using their container names as hostnames. In this case, the containers are not in the same network, as described earlier. So we first need to create a custom network:

```
docker network create custom_network
```

Next, we can assign both containers to the network and run them:

```
docker run -d --name backend --network custom_network -p 8080:8080 fastapi-app
docker run -d --name frontend --network custom_network -p 8051:8051 -e ENDPOINT_URL=http://backend:8080/classify streamlit-app
```

As we can see, the possibility to give the endpoint URL as an environment variable becomes handy; we don't need to rebuild the images if we need to change the endpoint URL. We will exploit this later on when deploying the containers to the Google Cloud. Though the containers run now on the same custom network, we can't use the loopback address for the same reasons as before: When the frontend container sends a request to `http://127.0.0.1:8080/classify`, it is essentially asking itself, "Do I have a service running on port 8080?" Since the FastAPI backend is in a separate container, the connection is refused. But when using the --name parameter and custom network, Docker assigns DNS names for the containers and takes care of the communication between them. This might feel difficult, but happily managing the Docker networking becomes easier with Docker Compose.

You might have encountered examples where the frontend container is, for example, a Node.js application, and it communicates with the backend container server via the `127.0.0.1` address. There is no separate Docker network, and the applications just use different ports of the same network. In this scenario, the communication between the frontend and backend happens via the browser, so the network isolation constraints do not necessarily apply. In this scenario, when the frontend is accessed, the browser downloads and executes the frontend code, and the requests are made from the browser, not the Node.js container. The backend container has to expose its `127.0.0.1:8080` to the host's loopback using `-p 127.0.0.1:8080:8080` (<host_address:host_port:<container_port>) for the browser to connect to it. This might sound complicated, but the key takeaway is that in this scenario, the communication happens via the browser, but in the Streamlit-FastAPI case, the frontend communicates with the container directly within its container.

Useful commands:

```
docker ps
docker ps -a
docker stop <container_id_or_name>
docker start <container_id_or_name>
docker rm <container_id_or_name>
docker image ls
docker rmi <image_id>
docker logs <container_id_or_name>
docker inspect <container_id_or_name>
docker exec <container_id_or_name> printenv (<VARIABLE>)
```

### Docker Compose

The networking between the containers can be confusing, and starting and stopping the containers can be tedious. Luckily, Docker Compose can take care of all this. Take a look at the `docker-compose.yml` file and run:

```
docker compose up
```

And that's it! Docker took care of starting up the containers and creating a network between them. Because the images were already built, we could use them and pass the port and endpoint URL values as environment variables when running Docker Compose.

We could also run the containers with `docker compose up -d` and stop them with `docker compose down`.

If we want to build and run the containers in one go, just run: `docker-compose up --build`.

This requires that we specify in the `docker-compose.yml` file that the images need to be built (`build:`) and where the Dockerfiles are (`context:` & `dockerfile:`).

So far, we have been able to access both the frontend and backend from the host (our browser). This is good for testing purposes but not necessarily something we want in real life. It could be preferable that the backend is accessible only from the frontend, meaning that the frontend is exposed to the host network, but the backend is exposed only to the private Docker network. This can be achieved by omitting the `ports:` section in the Docker Compose file (see `docker-compose.private.yml`). Omitting ports ensures the backend is not exposed to the host network, but it remains accessible to other containers in the same Docker network.

If the network driver is not specified, by default Docker creates a private internal network for the containers or services, Bridge network. The bridge driver network connects containers in the same network and isolates them from the external world unless explicitly exposed via the `ports` directive. Only the containers in the same network can communicate with each other. The containers can communicate with each other by their container names.

Besides the bridge, the other network drivers are `host` and `none`. When using `driver: host`, containers share the host network's interface and can be accessed directly from the host machine. With `driver: none`, the container won't have any network interface at all.

The `docker-compose.backend_network.yml` file uses the bridge network driver to specify a private shared backend network for the frontend and the backend. The functionality is the same as with the configuration used in the `docker-compose.private.yml` file.

We now have several Docker Compose files, so we want to give the file name as an argument to the Docker Compose command:

```
docker-compose -f docker-compose.private.yml up -d
docker-compose -f docker-compose.private.yml down
docker-compose -f docker-compose.private.yml up --build
```

### Tuning the Images

When running the containers locally for testing purposes, the image size is not a big issue. But when deploying the containers to the cloud, the size matters. The size affects performance and costs. Bigger images require more storage, memory, and processing time, which directly translates to costs in a cloud environment. I did some fine-tuning for the images, but not that much.

I chose `python:3.10-slim` as a base image for both containers. Its uncompressed on-disk size is around 145 MB. For Streamlit, I tried to install only the necessary libraries with pip but didn't do any deeper investigation into what was really needed. The final image size was 508 MB. Then I used a staged build and managed to reduce the size to 498 MB. The size could be reduced more by selecting some other base image and stripping out unnecessary libraries. But at least the image size was not in GBs, and I didn't want to take the Alpine Linux route, so I settled for this.

The FastAPI image was and is a bit more challenging. I have NVIDIA CUDA libraries in use, and the size of the pretrained Hugging Face PyTorch model is 328 MB. Using CUDA makes the inference faster, but the libraries take more space. The Hugging Face model can be saved locally or loaded on demand from the repository to the cache. If the model is saved locally and included in the image, it will make the image size bigger. If the model is loaded when the container is started, the startup process is slower and generates more network traffic. Likewise, CUDA makes the inference faster but requires more space and GPU.

I tried to figure out how all the different configuration options would affect the costs of deploying the containers to Google Cloud Run. And to be honest, I couldn't figure out the exact numbers.

The FastAPI image size was about 2.4 GB with the CUDA drivers and the model. When leaving CUDA out, the image size is reduced to 1.44 GB. The final image has the model in it, but no CUDA.

#### Non-root User

The services are run as root; the correct procedure would be to add a non-root user and start the container processes as it.

## Google Cloud Run Deployment

> **WARNING:** Following these instructions and deploying this project to Google Cloud Run will incur costs!

Cloud Run allows running containers in a fully managed serverless environment, scaling down to zero when there is no traffic. Google Cloud Run does not use docker-compose to manage or orchestrate multiple containers. Instead, each container is deployed as an independent service, which can communicate with each other over HTTP, enabling the replication of a multi-container docker-compose setup using multiple Cloud Run services. When deploying the containers to Cloud Run, they are hosted in separate environments, and each service (frontend and backend) will have a unique URL. Kubernetes is the appropriate tool if more sophisticated management and orchestration are needed.

Before we can do the deployment, we need to install Google Cloud CLI tools, enable billing, and activate the project in Google Cloud. Then we need the Google Container Registry (GCR) and Google Cloud Run services. I don't cover how all the steps are done, only the steps to build the containers and deploy them to the cloud. Almost everything can be done with the CLI, but for example, setting up the container registry and the project is more intuitive to do from the Google Cloud Console. It is good to get familiar with both of them anyway.

### Install Google Cloud CLI

Authenticate with your Google Cloud Account:

```
gcloud auth login
```

Make sure you have set your active project in Google Cloud. If you haven't already set it, do it with the following command:

```
gcloud config set project YOUR_PROJECT_ID
```

You can find your project ID in the Google Cloud Console, for example under the Cloud Run products page.

For example, my ```PROJECT_ID``` is  ```digit-prediction-test```.

### Build and Push Docker Images to Google Container Registry (GCR)

We could let Google handle the build process of the image. But let's do it manually, we will use basic docker commands.

Navigate to the directory containing the Dockerfile for the backend, and build the image with the correct tag. The image name is in the format ```REGISTRY/PROJECT_ID/IMAGE_NAME```.

```
docker build -t gcr.io/digit-prediction-test/backend-image .
```

Push the image to Google Container Registry:

```
docker push gcr.io/digit-prediction-test/backend-image
```

Navigate to the directory containing the Dockerfile for the frontend, and build the image with the correct tag:

```
docker build -t gcr.io/digit-prediction-test/frontend-image .
```

Push the image to Google Container Registry:

```
docker push gcr.io/digit-prediction-test/frontend-image
```

The containers can be seen in the Google Cloud Console's Artifact Registry.

### Deploy the Containers to Google Cloud Run

Containers can be deployed to the cloud with the commands below. When the command is run, the CLI will give a list of regions to choose from. I use the region ```europe-north1```. Or the list of options can be obtained with the command ```gcloud run regions list```. If all goes without issues, the URL of the service will be printed out.

Deploy first the backend:

```
gcloud run deploy backend --image gcr.io/digit-prediction-test/backend-image --platform managed --region europe-north1 --project=digit-prediction-test --allow-unauthenticated
```

```--allow-unauthenticated``` is needed to allow unauthenticated access to the service. Copy the backend's URL, which is in the format ```https://<backend-<hash>.<region>.run.app```, and add the endpoint path ```/classify``` to it. The endpoint URL can be given as an environment variable when deploying the frontend. Also, Google Cloud Run expects that the container listens on port ```8080```. This can be overridden with the ```PORT``` variable. The Streamlit port in the Dockerfile is set to ```8051```. If you need other environment variables, such as for configuring API keys or other settings, you can similarly add them using the ```--set-env-vars``` flag.

Deploy the frontend:

```
gcloud run deploy frontend --image gcr.io/digit-prediction-test/frontend-image --platform managed --region europe-north1 --project=digit-prediction-test --allow-unauthenticated --set-env-vars ENDPOINT_URL=https://<backend-<hash>.<region>.run.app/classify PORT=8080
```

### Test the Deployed Service

Once the frontend and backend are deployed to Google Cloud Run, you can visit the frontend URL to test the full functionality of your app:

```
https://<backend-<hash>.<region>.run.app/docs
https://<frontend-<hash>.<region>.run.app
```

### Restrict Public Access to the Backend

> **Note**: Service-to-Service authentication is not currently implemented in the frontend's source code. Running the commands below will block access to the backend!

As discussed with Docker Compose earlier, we might want to ensure that only the frontend is publicly accessible and the backend service is private, allowing only the frontend service to communicate with it.

#### Disable Public Access to the Backend

Update the backend's deployment to restrict unauthenticated access:

```
gcloud run services remove-iam-policy-binding backend --region europe-north1 --platform managed --member "allUsers" --role "roles/run.invoker"
```

When accessing the backend URL, you'll get "Error: Forbidden". Also, pressing the "Classify" button in the Streamlit app will result in a 403 error code from the backend.

The absence of "allUsers" can also be confirmed with the command:

```
gcloud run services get-iam-policy backend --region europe-north1 --platform managed
```

#### Create Custom Service Account

Create a service account named "frontend-sa" (can be any name), which will access the backend:

```
gcloud iam service-accounts create frontend-sa --description="Service account for frontend Cloud Run" --display-name="Frontend Service Account"
```

Ensure that the custom service account has the necessary ```roles/run.invoker``` permission to invoke the backend service:

```
gcloud run services add-iam-policy-binding backend --member="serviceAccount:frontend-sa@digit-prediction-test.iam.gserviceaccount.com" \
--role="roles/run.invoker" --platform managed --region europe-north1
```

#### Assign the Service Account to the Frontend Service:

```
gcloud run services update frontend --service-account=frontend-sa@digit-prediction-test.iam.gserviceaccount.com --platform managed --region europe-north1
```

---
TODO:
Add an Authorization header to the requests the frontend makes to the backend. This header will contain a JWT token that proves the frontend's identity.

```
pip install google-auth google-auth-httplib2
```

```
import google.auth
from google.auth.transport.requests import Request
import requests
...
```

Then in the frontend code:
* Retrieve the JWT token from the Google Cloud metadata server in the frontend.
* Include this token as a bearer token in your HTTP requests to the backend.
---

To verify if everything is working or not, check the logs:
```
gcloud services enable logging.googleapis.com
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=backend" --limit 10 --project digit-prediction-test
```

If the service account does not have the necessary rights, the logs will have something like this:

"textPayload: 'The request was not authenticated. Either allow unauthenticated invocations or set the proper Authorization header. Read more at <https://cloud.google.com/run/docs/securing/authenticating>"

To allow access again to the backend:

```
gcloud run services add-iam-policy-binding backend --member="allUsers" --role="roles/run.invoker" --platform managed --region europe-north1
```

### Debugging and Fixing Some Issues

Below are some issues you might encounter and how to debug and fix them.

You need to have an existing project and enable Google Container Registry (GCR). Also, you need to allow billing for Cloud Run. Your account won't be billed unless the actions stay in the free tier.

1. Ensure Docker is authenticated with Google Container Registry (GCR):

```
gcloud auth configure-docker
```

This command updates your Docker configuration to allow pushing images to GCR.

2. Ensure that you’re working in the correct Google Cloud project. Check your active projects:

```
gcloud config list project
```

If the project ID (e.g., digit-prediction-test) is not listed, set it:

```
gcloud config set project YOUR_PROJECT_ID
```

3. Ensure Your Project Has Container Registry Enabled. The Google Container Registry API must be enabled for your project. Enable it with this command:

```
gcloud services enable containerregistry.googleapis.com
```

4. Verify the Image Exists in GCR, or The image name or tag matches what you're referencing in your Cloud Run deployment:

```
gcloud container images list --repository=gcr.io/YOUR_PROJECT_ID
```

5. If there are old or unused images in GCR that might be causing confusion, consider deleting them:

```
gcloud container images delete gcr.io/PROJECT_ID/IMAGE_NAME
```

6. To find available regions:

```
gcloud run regions list
```

7. Ensure Your Container Listens on Port 8080. 

Cloud Run requires the container to listen on ```8080```. Update the backend's ```CMD``` instruction in the Dockerfile to reflect this.
Cloud Run automatically injects the ```PORT``` environment variable into the container. If your application doesn't explicitly set the port, ensure that it uses this variable.

8. View the logs for the failed deployment to diagnose the problem:

```
gcloud run services logs read SERVICE_NAME
```

### Suspend the Services (Scale to Zero) and Resume a Service (Scale Back Up)

According to Google's [documentation](https://cloud.google.com/run/docs/managing/services): "Cloud Run does not offer a direct way to make a service stop serving traffic, but you can achieve a similar result by revoking the permission to invoke the service to identities that are invoking the service. Notably, if your service is 'public', remove allUsers from the Cloud Run Invoker role (roles/run.invoker)."

To make the service no longer publicly accessible, revoke the roles/run.invoker permission for allUsers:

```
gcloud run services remove-iam-policy-binding SERVICE_NAME --member="allUsers" --role="roles/run.invoker" --platform managed --region REGION
```

To make the service public again, you can re-add the roles/run.invoker role for allUsers:

```
gcloud run services add-iam-policy-binding SERVICE_NAME --member="allUsers" --role="roles/run.invoker" --platform managed --region REGION
```

If the service is private, like in the case where the backend is blocked from the public, revoke the roles/run.invoker role for a service (or user) account:

```
gcloud run services remove-iam-policy-binding SERVICE_NAME --member="serviceAccount:SERVICE_ACCOUNT_EMAIL" --role="roles/run.invoker" --platform managed --region REGION
```

To restore the access:

```
gcloud run services add-iam-policy-binding SERVICE_NAME --member="serviceAccount:SERVICE_ACCOUNT_EMAIL" --role="roles/run.invoker" --platform managed --region REGION
```

#### Deleting the Service

```
gcloud run services delete SERVICE_NAME --platform managed --region REGION
```

Deleting the service does not automatically remove container images from Container Registry.

### GitHub Google Cloud Deployment Pipeline

TODO:

Implement deployments of containers to Google Cloud Run using a GitHub Actions deployment pipeline. Use Google "CONNECT REPO" functionality, or GitHub Actions with Docker or Google Cloud Build to build the images.

### Fine-Tuning the Inference Time

#### Optimize the PyTorch Model

* Quantization: Convert the model to a lower precision (e.g., int8) to reduce computation.
* Pruning: Remove unnecessary weights and neurons.
* Export to ONNX: Convert the model to ONNX format and use an optimized runtime like TensorRT.
* Test some other (smaller) or special MNIST models.

#### Docker Image Size Optimization

#### TorchScript and TorchServe

* Convert the model to a [TorchScript](https://pytorch.org/docs/stable/jit.html) or [TorchServe](https://pytorch.org/serve/) version for faster execution.
* Serve the model as REST API with TorchServe

#### GPU Support

* Use GPU in the model and leverage Cloud Run’s ability to use GPUs.
