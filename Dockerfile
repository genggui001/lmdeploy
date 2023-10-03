FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

RUN apt-get update && apt-get install -y wget git && rm -rf /var/lib/apt/lists/*

RUN wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda && rm Miniconda3-latest-Linux-x86_64.sh

RUN echo 'channels:\n\
  - defaults\n\
show_channel_urls: true\n\
default_channels:\n\
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main\n\
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r\n\
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2\n\
custom_channels:\n\
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
  pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud\n\
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud'\
> /root/.condarc

RUN /opt/miniconda/bin/pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /usr/src/app

COPY environment.yml ./

RUN /opt/miniconda/bin/conda env create -f ./environment.yml && /opt/miniconda/bin/conda clean --tarballs -y

ADD . /usr/src/app

EXPOSE 10045

CMD ["/opt/miniconda/envs/lmdeploy/bin/gunicorn", "-w", "16", "-b", "0.0.0.0:10045", "-t", "600", "lmdeploy.serve.api_server_on_tritonserver:app"]
