FROM computesoftware/zulu-openjdk-11:dev-utils-intel-mkl-2018.4-057-tools-deps-1.10.3.981-0e524d1

# clerk
EXPOSE 7777

# portal
EXPOSE 53755

# REPL
EXPOSE 8889

# REPL
EXPOSE 8050


COPY deps.edn /home/circleci/deps.edn

RUN echo '{:bind "0.0.0.0" :port 8889}' > ~/.nrepl.edn &&\
    cd /home/circleci && \
    clojure < /dev/null >&0

WORKDIR /home/circleci/
