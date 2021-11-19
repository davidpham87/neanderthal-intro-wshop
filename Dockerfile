FROM computesoftware/zulu-openjdk-11:dev-utils-intel-mkl-2018.4-057-tools-deps-1.10.3.981-0e524d1

EXPOSE 8889
EXPOSE 8080

COPY deps.edn /home/circleci/deps.edn

RUN echo '{:bind "0.0.0.0" :port 8888}' > ~/.nrepl.edn &&\
    cd /home/circleci && \
    clojure -M:repl < /dev/null >&0

WORKDIR /home/circleci/
