FROM pytorch/pytorch:latest
RUN apt update && apt install -y git wget ffmpeg libsm6 libxext6
RUN pip install packaging
RUN git clone https://github.com/NVIDIA/apex && cd apex && pip install -v --no-cache-dir ./
RUN git clone https://github.com/giandbt/synboost.git
RUN cd synboost && wget http://robotics.ethz.ch/~asl-datasets/Dissimilarity/models.tar && tar -xvf ./models.tar -C . && chmod -R 777 ./models && rm ./models.tar
COPY requirements.txt /workspace/synboost/requirements_demo.txt
RUN cd synboost && pip install -r /workspace/synboost/requirements_demo.txt
ENV TORCH_HOME=/opt/torch_home
RUN mkdir -p $TORCH_HOME
WORKDIR synboost
RUN mkdir tmp1 tmp2 && python demo.py --demo_folder tmp1 --save_folder tmp2 && rm -rf tmp1 tmp2 && chmod -R 777 $TORCH_HOME
COPY demo.py /workspace/synboost/main.py
RUN mkdir -p /workspace/synboost/results
CMD ["python", "main.py", "--demo_folder", "/input", "--save_folder", "/output"]
