FROM ubuntu:22.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
RUN apt-get update
RUN apt-get update && apt-get upgrade -y && apt-get install gcc g++ --yes
RUN apt-get update && apt-get install -y git
RUN apt-get update && apt-get install vim -y
RUN apt-get -y update && apt-get install unzip

RUN apt-get update && \
    apt-get install -y wget bzip2 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Anaconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/miniconda && \
    rm /tmp/miniconda.sh
ENV PATH=/opt/miniconda/bin:$PATH
RUN echo "source /opt/miniconda/etc/profile.d/conda.sh && conda activate base" >> /root/.bashrc

RUN apt-get update && apt-get clean

#RUN pwd && ls
#WORKDIR ~
#RUN pwd && ls

RUN mkdir -p /install  # Create directory before using WORKDIR
COPY install-step-1-by-conda.sh /install
WORKDIR /install
RUN bash -evx install-step-1-by-conda.sh ng                              # install all dependencies by conda

COPY . /app
WORKDIR /app
RUN conda run -n ng bash -evx install-step-2-for-feature-extraction.sh && conda install -y -n ng mhcflurry && mhcflurry-downloads fetch # download tools and database for extracting features (in TSV format) from neoepitope candidates

# This step downlaods a lot of data, so it is not run by default
# conda run -n ng sh -evx install-step-2-for-neoepitope-detection.sh # download tools and database for detecting neoepitope candidates (in FASTA format) from sequencing data

