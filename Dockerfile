FROM ubuntu:22.04

# Set environment variable for hnswlib
ENV HNSWLIB_NO_NATIVE=1

RUN apt-get update && \
    apt-get install -y wget gnupg curl unzip && \
    wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - && \
    echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list && \
    apt-get update && \
    apt-get install -y google-chrome-stable && \
    rm -rf /var/lib/apt/lists/*

# install chromedriver
RUN apt-get install -yqq unzip
RUN wget -O /tmp/chromedriver.zip http://chromedriver.storage.googleapis.com/`curl -sS chromedriver.storage.googleapis.com/LATEST_RELEASE`/chromedriver_linux64.zip
RUN unzip /tmp/chromedriver.zip chromedriver -d /usr/local/bin/



# remember to expose the port your app'll be exposed on.
EXPOSE 80

RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install -U pip

RUN pip install -U pip

COPY requirements.txt app/requirements.txt
RUN pip install -r app/requirements.txt

# copy into a directory of its own (so it isn't in the toplevel dir)
COPY . /app
# copy the classes folder
COPY Utils /app/Utils

WORKDIR /app

# run it!
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=80", "--server.address=0.0.0.0"]
# CMD ["app.py"]
# ENTRYPOINT ["python"]