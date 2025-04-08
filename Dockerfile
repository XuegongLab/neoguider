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

RUN mkdir -p /install
COPY env/freeze.env_export_no_builds.yml /install 
RUN conda env create --name ng --file /install/freeze.env_export_no_builds.yml

# Use the following if you would like to install the latest conda packages (caveat: this may not run successfully)
# CP install-step-1-by-conda.sh /install 
# WORKDIR /install
# RUN bash -evx install-step-1-by-conda.sh ng # install all dependencies by conda

COPY . /app
WORKDIR /app
RUN conda run -n ng bash -evx install-step-2-for-feature-extraction.sh # download tools and database for extracting features (in TSV format) from neoepitope candidates

# After running the above install-step-2-for-feature-extraction.sh,
# you still have to manually install the following, inside the docker container, to extract neoepitope features from peptide-MHC:
#  1 - netMHCpan and netMHCstabpan with the instructions from https://services.healthtech.dtu.dk/ (due to their license requirements)
#  2 - MHCflurry with ``conda install -y -n ng mhcflurry && mhcflurry-downloads fetch`` (because the automated installation does not work for some unknown reason)
# Once the manual installation is done, you can use the "docker commit" command to create a new image with the manually installed tools. 

# This step downloads about 60 GB of compressed data: the data involved are too big, so this step is not run by default
# conda run -n ng sh -evx install-step-2-for-neoepitope-detection.sh # download tools and database for detecting neoepitope candidates (in FASTA format) from sequencing data

