# Google Cloud GCE L4 Benchmark Steps

## Prepare environments
### 1. Create L4 GCE
```
create L4 resource
gcloud compute instances create YOUR_MACHINE_NAME --machine-type "g2-standard-8" --zone "us-central1-b" --maintenance-policy TERMINATE --restart-on-failure --boot-disk-size=200 --network-interface=network=YOUR_NETWORK_NAME

# create T4 resource
gcloud compute instances create t4-lsj --machine-type "n1-standard-8" --accelerator=count=1,type=nvidia-tesla-t4 --zone "us-central1-b" --maintenance-policy TERMINATE --restart-on-failure --boot-disk-size=200 --network-interface=network=llm-network

```
### 2. Install GPU driver
#### 2.1 link
- https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#debian
- https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Debian&target_version=11&target_type=deb_local

#### 2.2 check nvidia
```
lspci | grep -i nvidia
```

#### 2.3 check system
```
uname -m && cat /etc/*release
```

##### 2.4 install gcc

https://linuxize.com/post/how-to-install-gcc-compiler-on-debian-10/
```

sudo apt update
sudo apt install build-essential
sudo apt-get install manpages-dev
gcc --version
```

#### 2.5 install kernel headers
```
sudo apt-get install linux-headers-$(uname -r)
```
#### 2.6 install add-apt-repository
```
sudo apt update
sudo apt install software-properties-common
sudo apt update
```

#### 2.7 install wget
```
sudo apt-get install wget
```

#### 2.8 install cuda

https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#debian

https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Debian&target_version=11&target_type=deb_local

```
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-debian11-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-debian11-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo cp /var/cuda-repo-debian11-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo add-apt-repository contrib
sudo apt-get update
sudo apt-get -y install cuda
nvidia-smi
```

### 3. Install docker

https://docs.docker.com/engine/install/debian/
```
sudo apt-get update
sudo apt-get install \
    ca-certificates \
    curl \
    gnupg

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo docker run hello-world
```
### 4. Install nvidia docker
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)       && curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -       && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update

sudo apt-get install -y nvidia-docker2

sudo systemctl restart docker

docker run --gpus all -it  YOUR_DOCKER_IMAGE_NAME
```
### 5. [Optional] Install gcloud

https://cloud.google.com/sdk/docs/install#deb
```
sudo apt-get install apt-transport-https ca-certificates gnupg
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
sudo apt-get update && sudo apt-get install google-cloud-cli
sudo apt-get install google-cloud-cli
gcloud init
```

### 6. [Optional] Config Artifact Registry
```
gcloud auth configure-docker \
    us-central1-docker.pkg.dev
```
### 7. Run docker image
```
sudo docker run --gpus all -it nvcr.io/nvidia/pytorch:23.03-py3 /bin/bash
```

### 8. Install libraries and run test
```
git clone https://github.com/huggingface/diffusers.git
pip install --upgrade diffusers[torch]
pip install transformers
```
Write the test.py
```
from diffusers import DiffusionPipeline
import time
from diffusers import EulerAncestralDiscreteScheduler
import torch

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
pipeline.to("cuda")

batch_size = 1
prompt = ["An image of a squirrel in Picasso style"] * batch_size
generator = torch.Generator("cuda").manual_seed(123)
#warmup
for i in range(1,5):
    img = pipeline(prompt,
        num_inference_steps=20,
        guidance_scale=7.5,
        generator = generator,
    ).images[0]
#benchmark
start_time = time.time()
for i in range(1,20):
    img = pipeline(prompt,
        num_inference_steps=20,
        guidance_scale=7.5,
        generator = generator,
    ).images[0]
end_time = time.time()
period = end_time - start_time
print(float(period/20))
```
### 9. Run and monitor GPU
```
python test.py &
nvidia-smi
nvidia-smi -i
```

### 10. Benchmark

- docker: nvcr.io/nvidia/pytorch:23.03-py3
- diffusers library
- prompt: ”An image of a squirrel in Picasso style”
- parameters: model=runwayml/SD1.5, inf_step=20, cfg_scale=7.5, resolution=512x512, scheduler=Euler_A, seed=123
- GPU driver: 530.30.02
- CUDA: 12.1


|               |     T4       |      T4      |     L4       |     L4       |
|---------------|:------------:|-------------:|-------------:|-------------:|
|               | batch_size=1 | batch_size=4 | batch_size=1 | batch_size=4 |
|     time      |    9.3       |     34.79    |     3.017    |     11.23    |
|  utilization  |    100%      |     100%     |     100%     |     100%     |
|     power     |    70W       |     70W      |     70W      |     70W      |
|     memory    |    6.7GB     |     12GB     |     7GB      |     11.5GB   |
