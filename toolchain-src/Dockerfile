ARG DEBIAN_FRONTEND=noninteractive
FROM python:3.10.14

WORKDIR /root

RUN echo "### Disabled."
RUN echo "### Support packages:"

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
RUN apt-get update -qq -y --fix-missing \
 && apt-get install -qq -y gcc g++ curl make xxd wget apt-utils dos2unix software-properties-common git autoconf \
   automake libtool python3 unzip cmake build-essential protobuf-compiler libprotoc-dev \
   ffmpeg libsm6 libxext6  > /dev/null

COPY requirements.txt /root/requirements.txt
COPY setup.py /root/setup.py
COPY conversion_block /root/conversion_block
COPY src/run-block.sh /usr/local/bin/conversion-run-block

ENV PATH="/usr/local/bin:${PATH}"
RUN echo "### Installing Python dependencies:" \
 && pip install -r requirements.txt \
 && echo " ###> Copying compile script to dockerfile" \
 && dos2unix /usr/local/bin/conversion-run-block \
 && chmod +x /usr/local/bin/conversion-run-block

ENTRYPOINT ["bash", "/usr/local/bin/conversion-run-block"]