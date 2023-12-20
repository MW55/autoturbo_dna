FROM pytorch/pytorch

ENV DEBIAN_FRONTEND=noninteractive

COPY . /autoturbo_dna
WORKDIR /autoturbo_dna

RUN apt-get update && \
    apt-get install -y \
        git \
        python3-pip \
        python3-dev \
        python3-opencv \
        libglib2.0-0

RUN python3 -m pip install --upgrade pip

RUN pip3 install torch numpy scipy regex pytorch-ignite

CMD [ "bash" ]
