#!/bin/bash
/Users/dixantmittal/Applications/kafka_2.12-1.1.0/bin/zookeeper-server-start.sh /Users/dixantmittal/Applications/kafka_2.12-1.1.0/config/zookeeper.properties &
/Users/dixantmittal/Applications/kafka_2.12-1.1.0/bin/kafka-server-start.sh /Users/dixantmittal/Applications/kafka_2.12-1.1.0/config/server.properties && fg
