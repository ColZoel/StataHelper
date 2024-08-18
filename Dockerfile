FROM python:3.11-buster
LABEL authors="collinzoeller"

RUN mkdir "/statahelper/"
COPY . "/statahelper/"
FROM continuumio/miniconda3
ADD ./environment.yaml /statahelper/environment.yaml
RUN conda env create -f /statahelper/environment.yaml
RUN echo "source activate env" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH
RUN echo "source activate $(head -1 /statahelper/environment.yaml | cut -d' ' -f2)" > ~/.bashrc
ENV PATH /opt/conda/envs/$(head -1 /statahelper/environment.yaml | cut -d' ' -f2)/bin:$PATH
RUN pip install --upgrade pip
RUN pip install -e .
RUN pip3 install pytest

WORKDIR "/statahelper/"

CMD "pytest"
ENV PYTHONDONTWRITEBYTECODE=true
