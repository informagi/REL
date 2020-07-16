FROM pytorch/pytorch

# download necessary files
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    tar

# default year is 2019
ARG WIKI_YEAR="2019"

# make sure the year is 2014/2019; if not quit, as the downloads won't work
RUN test "$WIKI_YEAR" = "2019" || test "$WIKI_YEAR" = "2014"

# these files are generic for any wikipedia version
RUN wget -q "http://gem.cs.ru.nl/generic.tar.gz" && \
    tar xzf "generic.tar.gz" && \
    rm "generic.tar.gz"

# install pip and REL
RUN conda install -y pip
RUN git clone "https://github.com/informagi/REL" && \
    cd REL && \
    pip install -e . && \
    cd ..

# pull files necessary for WIKI_YEAR (2014 or 2019) English wiki setup from fileserver;
# do this last to optimize cache usage
RUN wget -q "http://gem.cs.ru.nl/wiki_$WIKI_YEAR.tar.gz" && \
    tar xzf "wiki_$WIKI_YEAR.tar.gz" && \
    rm "wiki_$WIKI_YEAR.tar.gz"


# expose the API port
EXPOSE 5555

# run REL server
ENTRYPOINT python -m REL.server ./ "wiki_$WIKI_YEAR" --bind 0.0.0.0 --port 5555
