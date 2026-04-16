## General Idea

### Agent Flow

```
User Query
    │
    ▼
┌─────────────┐
│  Supervisor │  ← (i)
└──────┬──────┘
       │
  ┌────┴─────────────────────┐
  │                          │
  ▼                          ▼
Out of Scope         Parallel Processor 
  │                 ┌────────┴────────┐
  │                 ▼                 ▼
  │            KB Agent          SQL Agent        ← (ii)
  │         (document data)   (structured data)
  │                 └────────┬────────┘
  │                          ▼
  │                     Synthesize
  │                          │
  └──────────────────────────┤
                             ▼
                      Final Response

                      **Note**:
                      (i): Classifies intent, decides which agents to invoke
                      (ii): The number Agent of parallel process can be flexible depend on type of information the question need to query.

```

### Project Structure
```
stock-analysis-agent/
├── app/
│   ├── main.py                      # Entry point & configuration
│   ├── agentic_bot.py               # Graph builder & orchestrator
│   ├── core/
│   │   └── state.py                 # Shared state schema (TypedDict)
│   ├── nodes/
│   │   ├── supervisor.py            # Intent classification node
│   │   ├── kb_agent.py              # AWS Bedrock KB agent
│   │   ├── sql_agent.py             # AWS Redshift SQL generation & execution node
│   │   ├── parallel_processor.py    # Runs KB + SQL agents concurrently
│   │   ├── synthesize.py            # Final answer generation node
│   │   └── out_of_scope.py          # Out-of-scope response handler
│   └── prompts/
│       ├── technical_analysis.txt   # System prompt for SQL agent (price/technical data)
│       └── financial_report.txt     # System prompt for SQL agent (financial reports)
├── preparation/
│   └── curated_unstructured.py      # S3/Langfuse/Bedrock processing Lambda
├── requirements.txt
└── README.md
```

## Setup

### 1. Self-host Langfuse via Docker Compose

Follow this [link](https://langfuse.com/self-hosting/deployment/docker-compose): https://langfuse.com/self-hosting/deployment/docker-compose

### 2. Create Langfuse API Key

- Log in to Langfuse: Access your Langfuse account on the official website.
- Navigate to Project: If you haven't already, create a new project or select an existing one.
- Access Project Settings: Within your chosen project, locate and click on "Settings" in the left-hand sidebar.
- Create API Keys: In the Settings section, find the "API Keys" or "Create API Keys" option and click on it.
- Copy and Save Credentials: Langfuse will generate a set of API credentials, typically including a Secret Key, Public Key, and the Host URL. Copy these values and store them securely, as the Secret Key is often only viewable once.

Reference [link](https://langfuse.com/faq/all/where-are-langfuse-api-keys): https://langfuse.com/faq/all/where-are-langfuse-api-keys

### 3. Setup Python Virtual Environment

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Setup Credentials
```sh
export LANGFUSE_SECRET_KEY=""
export LANGFUSE_PUBLIC_KEY=""
export LANGFUSE_HOST=""

export AWS_PROFILE=''
export AWS_DEFAULT_REGION='us-east-1'
export AWS_REGION='us-east-1'
```

### 5. Prepare Knowledge Bases and Redshift configurations

From `main.py`, enter your configurations:
```json
        "kb_agent": {
            "market_information": {
                "id": "<enter KB ID here>",
                "name": "market_information",
                ...
            },
            "company_information": {
                "id": "<enter KB ID here>",
                "name": "company_information",
                ...
            }
        },
        "sql_agent": {
            "technical_analysis": {
                "database_name": "<enter database name>",
                "workgroup_name": "<enter workgroup name>",
                "secret_arn": "<enter secret ARN>",
                "target_schema": "<enter target schema>",
                "db_region": "us-east-1",
                ...
            },
            "financial_report": {
                "database_name": "<enter database name>",
                "workgroup_name": "<enter workgroup name>",
                "secret_arn": "<enter secret ARN>",
                "target_schema": "<enter target schema>",
                "db_region": "us-east-1",
                ...
            }
```

### 6. Enter user query

From `main.py`, enter the user's query:
```python
    query = "Your query here"
```

### 7. Execute Agentic Bot
```sh
python -m app.main
```
