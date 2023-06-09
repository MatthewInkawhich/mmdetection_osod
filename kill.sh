#!/bin/bash


kill $(ps aux | grep "inkawhmj/mmdet/bin" | grep -v grep | awk '{print $2}')
kill $(ps aux | grep "test" | grep -v grep | awk '{print $2}')
kill $(ps aux | grep "train" | grep -v grep | awk '{print $2}')
