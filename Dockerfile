FROM tensorflow/tensorflow:latest-gpu

EXPOSE 80

RUN apt-get update && yes Y | apt-get install python python-tk idle python-pmw python-imaging
RUN pip install -U pip

# copy over our requirements.txt file
COPY . /cavelab/
RUN pip install -r /cavelab/requirements.txt
RUN python /cavelab/setup.py develop
