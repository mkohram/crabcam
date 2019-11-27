#!/usr/bin/env bash
# should be installed as a cron job for root

date -s "$(curl -s --head http://google.com | grep ^Date: | sed 's/Date: //g')"
