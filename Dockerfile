FROM pytorch/pytorch

# download necessary files
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    curl \
    git \
    libsqlite3-0 \
    libsqlite3-dev \
    sqlite3 \
    tar \
    vim

# install pip and REL
RUN conda install -y pip
RUN git clone "https://github.com/informagi/REL" && \
    cd REL && \
    pip install -e . && \
    cd ..

# ????
RUN chmod -R 777 /workspace && chown -R root:root /workspace

# expose the API port
EXPOSE 5555

CMD ["bash", "-c"]
