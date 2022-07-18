FROM python:3.8-bullseye

ARG UID
RUN mkdir -p /home/annclonevalidation
RUN useradd -u ${UID} annclonevalidation 

RUN apt-get update \
 && apt-get install -y --no-install-recommends  \
        build-essential \
        debhelper \
        devscripts \
        gcc \
        gettext \
        libffi-dev \
        libjpeg-dev \
        libmemcached-dev \
        libpq-dev \
        libxml2 \
        libxml2-dev \
        libxslt1-dev \
        memcached \
        netcat \
        python3-dev \
        python3-gdal \
        python3-ldap \
        python3-lxml \
        python3-pil \
        python3-pip \
        python3-psycopg2 \
        zip \
        zlib1g-dev \
        default-jre \
        default-jdk \
        python \
        python-tk \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /home/annclonevalidation
COPY . .

# separate venv for python2
RUN curl https://bootstrap.pypa.io/pip/2.7/get-pip.py --output /home/get-pip.py
RUN python2 /home/get-pip.py
RUN python2 -m pip install virtualenv
RUN python2 -m virtualenv /home/.venvpy2
RUN /home/.venvpy2/bin/pip install --upgrade pip
RUN /home/.venvpy2/bin/pip install -r requirements.txt

RUN chown -R annclonevalidation ./
RUN chown -R annclonevalidation:annclonevalidation ../.venvpy2

USER annclonevalidation
WORKDIR /home/annclonevalidation