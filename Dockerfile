FROM continuumio/miniconda3:latest
ENV LANG=C.UTF-8

# Install base libraries
RUN apt update -y 
RUN apt install -y wget nano htop curl libgl1

# Create working environment (CUDA, python version reconfigure needed)
ENV CONDA_ENV_NAME yolov8conv
WORKDIR /app
COPY . .
RUN conda env create --name $CONDA_ENV_NAME --file environment.yml
RUN echo "conda activate $CONDA_ENV_NAME" >> ~/.bashrc

# House-keeping
RUN conda clean --all -y
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN apt autoclean
RUN apt autoremove

CMD ["/bin/bash"]
