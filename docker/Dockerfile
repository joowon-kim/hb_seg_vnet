
FROM tensorflow/tensorflow:2.2.0-gpu-jupyter

RUN apt update -y && apt upgrade -y
RUN apt install git vim -y
RUN pip install nibabel
RUN pip install keras==2.4.3
RUN pip install scikit-image

RUN mkdir -p /home/user
RUN chmod ugo+rwX /home/user
ENV HOME /home/user
ENV DATA /data

COPY run_notebook.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/run_notebook.sh

VOLUME ["/data"]
VOLUME ["/code"]

WORKDIR /data
CMD /code/run-app

