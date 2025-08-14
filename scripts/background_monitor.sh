#!/bin/bash
# Background monitoring script that checks every 5 minutes

while true; do
    echo "========================="
    echo "Batch Status Check: $(date)"
    echo "========================="
    python scripts/check_batch_status.py
    
    # Check if completed
    para_status=$(python -c "
import json
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
with open('data/work/paraphrase_batch_info.json') as f:
    info = json.load(f)
batch = client.batches.retrieve(info['batch_id'])
print(batch.status)
")
    
    summary_status=$(python -c "
import json
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
with open('data/work/summary_batch_info.json') as f:
    info = json.load(f)
batch = client.batches.retrieve(info['batch_id'])
print(batch.status)
")
    
    if [[ "$para_status" == "completed" ]] && [[ "$summary_status" == "completed" ]]; then
        echo "✅ BOTH BATCHES COMPLETED!"
        # Download results
        python scripts/download_batch_results.py
        break
    fi
    
    # Wait 5 minutes
    sleep 300
done