FROM solr:9.5

USER root

# Install SQL dependencies
RUN apt-get update && \
    apt-get install -y wget unzip && \
    mkdir -p /opt/solr/contrib/sql && \
    cd /opt/solr/contrib/sql && \
    wget https://repo1.maven.org/maven2/org/apache/solr/solr-sql/9.5.0/solr-sql-9.5.0.jar && \
    wget https://repo1.maven.org/maven2/org/apache/calcite/calcite-core/1.35.0/calcite-core-1.35.0.jar && \
    wget https://repo1.maven.org/maven2/org/apache/calcite/calcite-linq4j/1.35.0/calcite-linq4j-1.35.0.jar && \
    wget https://repo1.maven.org/maven2/org/apache/calcite/avatica/avatica-core/1.23.0/avatica-core-1.23.0.jar && \
    wget https://repo1.maven.org/maven2/com/google/protobuf/protobuf-java/3.21.7/protobuf-java-3.21.7.jar && \
    wget https://repo1.maven.org/maven2/org/apache/calcite/avatica/avatica-metrics/1.23.0/avatica-metrics-1.23.0.jar && \
    wget https://repo1.maven.org/maven2/org/locationtech/jts/jts-core/1.19.0/jts-core-1.19.0.jar && \
    wget https://repo1.maven.org/maven2/org/codehaus/janino/janino/3.1.9/janino-3.1.9.jar && \
    wget https://repo1.maven.org/maven2/org/codehaus/janino/commons-compiler/3.1.9/commons-compiler-3.1.9.jar && \
    cp *.jar /opt/solr/server/solr-webapp/webapp/WEB-INF/lib/ && \
    chown -R solr:solr /opt/solr/contrib/sql /opt/solr/server/solr-webapp/webapp/WEB-INF/lib/*.jar

USER solr 