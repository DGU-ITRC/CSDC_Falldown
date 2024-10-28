#!/bin/bash
nohup uvicorn server:app --host 0.0.0.0 --port 50004 --reload