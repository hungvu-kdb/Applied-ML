import logging
import boto3
import asyncio
from typing import Dict, Any
from http import HTTPStatus
from datetime import datetime

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def get_value_by_key(parameters, key_name):
    for param in parameters:
        if param["name"] == key_name:
            return param["value"]
    return None

async def async_retrieve(bedrock_agent_runtime, KB_ID, inputText, retrievalConfiguration):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: bedrock_agent_runtime.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={'text': inputText},
            retrievalConfiguration=retrievalConfiguration
        )
    )

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler for processing Bedrock agent requests.
    
    Args:
        event (Dict[str, Any]): The Lambda event containing action details
        context (Any): The Lambda context object
    
    Returns:
        Dict[str, Any]: Response containing the action execution results
    
    Raises:
        KeyError: If required fields are missing from the event
    """
    try:
        KB_ID = "8EXRPHDCFC"
        action_group = event['actionGroup']
        function = event['function']
        message_version = event.get('messageVersion',1)
        inputText = event.get('inputText','')
        parameters = event.get('parameters', [])

        industry = get_value_by_key(parameters, "industry")
        year = get_value_by_key(parameters, "year")
        stock_code = get_value_by_key(parameters, "stock_code")

        if not inputText:
            return {
                'response': {
                    'actionGroup': action_group,
                    'function': function,
                    'functionResponse': {
                        "responseState": "REPROMPT",
                        'responseBody': {
                            'TEXT': {
                                'body': 'Please provide query to search information'
                            }
                        }
                    },
                    'messageVersion': message_version
                }
            }

        bedrock_agent_runtime = boto3.client('bedrock-agent-runtime')

        if not year:
            now = datetime.now()
            year = int(now.year)

        retrievalConfiguration={
            "vectorSearchConfiguration": {
                "numberOfResults":5,
                "implicitFilterConfiguration": {
                    "metadataAttributes":[
                        {
                            "key": "year",
                            "type": "NUMBER",
                            "description": f"The year in which the document/report is about. If not mention in the query, use the current year: {year} as value"
                        },
                        {
                            "key": "source_company",
                            "type": "STRING",
                            "description": "The company name that publish the document/report. Possible values include ['acbs', 'kbs', 'masvn', 'mbs', 'vietcap', 'vpb', 'yuanta']. If not mention in the query, can skip this field."
                        },
                        {
                            "key": "stock_code",
                            "type": "STRING",
                            "description": "The ticker name of the company. If not mention in the query, skip it"
                        },
                        {
                            "key": "industry",
                            "type": "STRING",
                            "description": "the industry in which the company operates. Possible values include ['retail', 'real_estate', 'oil_gas', 'tourism', 'household_goods', 'industrial_services', 'chemicals', 'banking', 'automotive', 'resources', 'food', 'utilities', 'telecommunications', 'construction', 'healthcare', 'insurance', 'finance', 'information_technology', 'media']."
                        },
                    ],
                    "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-5-sonnet-20240620-v1:0"
                },
            } 
        }

        # Use asyncio to run the retrieve call asynchronously
        loop = asyncio.get_event_loop()
        response = loop.run_until_complete(
            async_retrieve(bedrock_agent_runtime, KB_ID, inputText, retrievalConfiguration)
        )

        # Process and return the results in structured JSON
        results = []
        for result in response.get('retrievalResults', []):
            results.append({
                'content': result.get('content', {}).get('text', '') or result.get('metadata', {}).get('x-amz-bedrock-kb-description', ''),
                'source': result.get('location', {}).get('s3Location', {}).get('uri', ''),
                'page': result.get('metadata', {}).get('x-amz-bedrock-kb-document-page-number', '')
            })

        response_body = {
            'results': results,
            'count': len(results)
        }
        action_response = {
            'actionGroup': action_group,
            'function': function,
            'functionResponse': {
                'responseBody': response_body
            }
        }
        response = {
            'response': action_response,
            'messageVersion': message_version
        }

        logger.info('Response: %s', response)
        return response

    except KeyError as e:
        logger.error('Missing required field: %s', str(e))
        return {
            'statusCode': HTTPStatus.BAD_REQUEST,
            'body': f'Error: {str(e)}'
        }
    except Exception as e:
        logger.error('Unexpected error: %s', str(e))
        return {
            'statusCode': HTTPStatus.INTERNAL_SERVER_ERROR,
            'body': 'Internal server error'
        }
