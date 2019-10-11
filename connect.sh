#!/usr/bin/env bash

POST_ARGS="user=ahalogy%40quotient.com&password=${GPASS}&url=https%3A%2F%2Fwww.coupons.com&cmd=authenticate&Login=Log+In"
POST_URL="http://captiveportal-login.corp.coupons.com/cgi-bin/login"
IP_FILE="/tmp/ip"

>&2 echo "Allowing the guest network to connect (waiting for 30 seconds)"
sleep 30

while true; do
  curl -s 'https://en.wikipedia.org/w/api.php?action=query&format=json' | grep batchcomplete &> /dev/null

  if [ $? -ne 0 ]; then
    >&2 echo "Attempting to reconnect ..."
    /usr/bin/curl -d "${POST_ARGS}" -X POST "${POST_URL}" &> /dev/null
  else
    >&2 echo "Internet was available"
  fi
  
  cmp --silent "${IP_FILE}" <(ifconfig wlan0 | grep 'inet ' | tr -s ' ' | cut -d' ' -f3)

  if [ $? -ne 0 ]; then
    >&2 echo 'IP has changed. broadcasting.'
    new_ip=`ifconfig wlan0 | grep 'inet ' | tr -s ' ' | cut -d' ' -f3`
    echo "${new_ip}" > ${IP_FILE}
    curl -sX POST -H 'Content-type: application/json' --data "{\"text\": \"Hermes believes he's at: ${new_ip}\", \"icon_emoji\": \":hermes:\"}" ${SLACK_HOOK} &> /dev/null
  fi

  sleep 5
done
