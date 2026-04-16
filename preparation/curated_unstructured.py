import boto3
import json
import os
import time
import logging
import asyncio
from botocore.exceptions import ClientError
from utils.langfuse_utility import get_langfuse_client
from utils.prompt_loader import PromptLoader
from langfuse import observe

logger = logging.getLogger()
logger.setLevel(logging.INFO)

LANGFUSE_FLUSH_TIME_SLEEP = 30
ENVIRONMENT = os.environ.get('ENVIRONMENT')
PROMPT_NAME = os.environ.get('PROMPT_NAME')
os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = ENVIRONMENT


@observe(name="smart_split_text", capture_input=False, capture_output=False)
def smart_split_text(text, max_tokens=90000, chars_per_token=4, langfuse_client=None):
    """
    Chia text thông minh theo đoạn văn để tránh cắt giữa câu
    
    Args:
        text (str): Nội dung cần chia
        max_tokens (int): Số token tối đa mỗi phần
        chars_per_token (int): Ước tính số ký tự trên 1 token
    
    Returns:
        list: Danh sách các phần text
    """
    if langfuse_client:
        langfuse_client.update_current_span(
            input={
                "text": text,
                "max_tokens": max_tokens,
                "chars_per_token": chars_per_token,
            }
        )
    max_chars = max_tokens * chars_per_token
    chunks = []
    current_chunk = ""
    
    # Chia theo đoạn văn
    paragraphs = text.split('\n\n')
    
    for paragraph in paragraphs:
        # Nếu thêm đoạn này vào chunk hiện tại mà vẫn không vượt quá giới hạn
        if len(current_chunk) + len(paragraph) + 2 <= max_chars:
            if current_chunk:
                current_chunk += '\n\n' + paragraph
            else:
                current_chunk = paragraph
        else:
            # Lưu chunk hiện tại và bắt đầu chunk mới
            if current_chunk:
                chunks.append(current_chunk)
            
            # Nếu đoạn văn quá dài, chia nhỏ hơn
            if len(paragraph) > max_chars:
                sentences = paragraph.split('. ')
                temp_chunk = ""
                for sentence in sentences:
                    if len(temp_chunk) + len(sentence) + 2 <= max_chars:
                        if temp_chunk:
                            temp_chunk += '. ' + sentence
                        else:
                            temp_chunk = sentence
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk)
                        temp_chunk = sentence
                if temp_chunk:
                    current_chunk = temp_chunk
            else:
                current_chunk = paragraph
    
    # Thêm chunk cuối cùng
    if current_chunk:
        chunks.append(current_chunk)
    
    if langfuse_client:
        langfuse_client.update_current_span(
            output={
                "chunks": chunks,
            }
        )
    return chunks

@observe(name="get_bedrock_client", capture_input=False, capture_output=False)
def get_bedrock_client(llm_config, region, langfuse_client=None):
    from botocore.config import Config
    
    config = Config(read_timeout=2000)
    cross_account_role_arn = llm_config.get('cross_account_role_arn')
    
    if cross_account_role_arn:
        logger.info(f"Assuming cross-account role for Bedrock access")
        sts_client = boto3.client('sts')
        assumed_role = sts_client.assume_role(
            RoleArn=cross_account_role_arn,
            RoleSessionName=f"process-bcpt-{int(time.time() * 1000)}",
            DurationSeconds=3600
        )
        creds = assumed_role['Credentials']
        
        bedrock_client = boto3.client(
            'bedrock-runtime',
            aws_access_key_id=creds['AccessKeyId'],
            aws_secret_access_key=creds['SecretAccessKey'],
            aws_session_token=creds['SessionToken'],
            region_name=llm_config.get('model_region', region),
            config=config
        )
        logger.info(f"Created Bedrock session with unique ID")
    else:
        bedrock_client = boto3.client(
            'bedrock-runtime',
            region_name=llm_config.get('model_region', region),
            config=config
        )
        logger.info("Using default Bedrock client (no cross-account role)")
    
    return bedrock_client

@observe(name="load_prompt", capture_input=False, capture_output=False)
def load_prompt(langfuse_client):
    """Load prompt from Langfuse"""
    prompt_loader = PromptLoader(langfuse_client)
    label = ENVIRONMENT if ENVIRONMENT else "latest"
    if langfuse_client:
        langfuse_client.update_current_span(
            input={
                "prompt_name": PROMPT_NAME,
                "label": label,
            }
        )
    
    prompt_obj = asyncio.run(prompt_loader.get_prompt(PROMPT_NAME, label))
    llm_config = prompt_obj.config.get("llm_config", {})
    
    if langfuse_client:
        langfuse_client.update_current_span(
            output={
                "llm_config": llm_config,
            }
        )
    return prompt_obj, llm_config

@observe(name="invoke_bedrock_model", as_type="generation", capture_input=False)
def invoke_bedrock_model(compiled_prompt, llm_config, region, langfuse_client=None, prompt_obj=None):
    """Invoke Bedrock model and capture metrics"""
    bedrock = get_bedrock_client(llm_config, region, langfuse_client)
    
    messages = [{"role": "user", "content": [{"text": compiled_prompt}]}]
    
    model_id = llm_config.get('inference_profile_arn') or llm_config.get('model_id', os.environ.get('BEDROCK_MODEL_ID'))
    
    inference_config = {
        "maxTokens": llm_config.get('max_tokens', 20000)
    }
    
    sampling_params = llm_config.get('sampling_params')
    if sampling_params is None:
        if 'temperature' in llm_config:
            inference_config['temperature'] = llm_config['temperature']
        if 'top_p' in llm_config:
            inference_config['topP'] = llm_config['top_p']
    else:
        if 'temperature' in sampling_params and 'temperature' in llm_config:
            inference_config['temperature'] = llm_config['temperature']
        if 'top_p' in sampling_params and 'top_p' in llm_config:
            inference_config['topP'] = llm_config['top_p']
    
    if langfuse_client:
        update_params = {
            "model": llm_config.get('model_id', model_id),
            "input": compiled_prompt,
            "metadata": inference_config
        }
        if prompt_obj:
            update_params["prompt"] = prompt_obj
        langfuse_client.update_current_generation(**update_params)
    
    response = bedrock.converse(
        modelId=model_id,
        messages=messages,
        inferenceConfig=inference_config
    )
    
    claude_response = response['output']['message']['content'][0]['text'].strip()
    
    if langfuse_client:
        usage = response.get('usage', {})
        langfuse_client.update_current_generation(
            output=claude_response,
            usage_details={
                "input_tokens": usage.get('inputTokens', 0),
                "output_tokens": usage.get('outputTokens', 0),
            }
        )
    
    return claude_response

@observe(name="phan_tich_bao_cao_with_retry", capture_input=False)
def phan_tich_bao_cao_with_retry(file_content, region='us-east-1', max_retries=5, langfuse_client=None):
    """
    Wrapper function để retry khi gặp throttling
    """
    if langfuse_client:
        langfuse_client.update_current_span(
            input={"region": region, "max_retries": max_retries}
        )
    for attempt in range(max_retries):
        try:
            return phan_tich_bao_cao(file_content, region, langfuse_client)
        except Exception as e:
            error_str = str(e)
            # Kiểm tra nếu là throttling error
            if 'ThrottlingException' in error_str or 'throttled' in error_str.lower() or 'rate limit' in error_str.lower():
                if attempt < max_retries - 1:
                    wait_time = 60
                    logger.warning(f"Gặp throttling, chờ {wait_time}s trước khi retry (lần {attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Đã retry {max_retries} lần nhưng vẫn bị throttling")
                    raise
            else:
                # Nếu không phải throttling error thì throw luôn
                raise

@observe(name="phan_tich_bao_cao", capture_input=False)
def phan_tich_bao_cao(file_content, region='us-east-1', langfuse_client=None):
    """Nhận đoạn text kết hợp với prompt để gởi đến Bedrock Claude Sonnet 3.7"""
    if langfuse_client:
        langfuse_client.update_current_span(
            input={"region": region, "content_length": len(file_content)}
        )
    
    try:
        prompt_obj, llm_config = load_prompt(langfuse_client)
    except Exception as e:
        logger.error(f"Failed to load prompt from Langfuse: {e}")
        raise
    
    compiled_prompt = prompt_obj.compile(noi_dung_bao_cao=file_content)
    claude_response = invoke_bedrock_model(compiled_prompt, llm_config, region, langfuse_client, prompt_obj)
    
    return claude_response

@observe(name="read_s3_file", capture_input=False)
def read_s3_file(bucket_name, object_key, region='us-east-1', langfuse_client=None):
    """
    Đọc nội dung file từ S3
    
    Args:
        bucket_name (str): Tên S3 bucket
        object_key (str): Key của object trong S3
        region (str): AWS region
    
    Returns:
        str: Nội dung file
    """
    if langfuse_client:
        langfuse_client.update_current_span(
            input={
                "bucket_name": bucket_name,
                "object_key": object_key,
                "region": region,
            }
        )
    s3_client = boto3.client('s3', region_name=region)
    
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        file_content = response['Body'].read().decode('utf-8')
        return file_content
    except ClientError as e:
        print(f"Lỗi khi đọc file {object_key} từ bucket {bucket_name}: {e}")
        return None

@observe(name="process_s3_file", capture_input=False, capture_output=False)
def process_s3_file(bucket_name, object_key, max_tokens=90000, region='us-east-1', langfuse_client=None):
    """
    Đọc file từ S3, chia thành nhiều phần
    
    Args:
        bucket_name (str): Tên S3 bucket
        object_key (str): Key của object trong S3
        max_tokens (int): Số token tối đa mỗi phần
        region (str): AWS region
    
    Returns:
        list: Danh sách các chunks
    """
    if langfuse_client:
        langfuse_client.update_current_span(
            input={
                "bucket_name": bucket_name,
                "object_key": object_key,
                "max_tokens": max_tokens,
                "region": region,
            }
        )
    # Đọc nội dung file từ S3
    file_content = read_s3_file(
        bucket_name=bucket_name,
        object_key=object_key,
        region=region,
        langfuse_client=langfuse_client,
    )
    
    if file_content is None:
        return []
    
    # Chia file thành nhiều phần
    chunks = smart_split_text(file_content, max_tokens, langfuse_client=langfuse_client)
    
    if langfuse_client:
        langfuse_client.update_current_span(
            output={
                "chunks": chunks,
            }
        )
    return chunks

@observe(name="split_result_and_write", capture_input=False, capture_output=False)
def split_result_and_write(s3_client, bucket_name, result, output_filename, langfuse_client):
    """
    Chia result thành 2 phần và ghi ra 2 file riêng biệt
    """
    if langfuse_client:
        langfuse_client.update_current_span(
            input={"bucket_name": bucket_name, "result": result, "output_filename": output_filename}
        )
    # Tìm vị trí của tag <metadata>
    metadata_start = result.find('<metadata>')
    metadata_end = result.find('</metadata>') + len('</metadata>')
    
    if metadata_start == -1 or metadata_end == -1:
        print("Không tìm thấy tag metadata trong kết quả")
        return False, False
    
    # Phần 1: từ đầu đến trước tag <metadata>
    noi_dung_bao_cao = result[:metadata_start].strip()
    
    # Phần 2: nội dung bên trong tag <metadata> (bỏ tag)
    metadata_content = result[metadata_start + len('<metadata>'):metadata_end - len('</metadata>')].strip()
    
    # Tạo tên file cho 2 phần
    output_key1 = f"knowledge_base_cac_loai_bao_cao_curated/{output_filename}"
    output_key2 = f"knowledge_base_cac_loai_bao_cao_curated/{output_filename}.metadata.json"
    
    # Ghi phần 1
    success1 = write_output_to_s3(s3_client, bucket_name, output_key1, noi_dung_bao_cao, langfuse_client)
    
    # Ghi phần 2
    success2 = write_output_to_s3(s3_client, bucket_name, output_key2, metadata_content, langfuse_client)
    
    if langfuse_client:
        langfuse_client.update_current_span(
            output={
                "txt_file_content": success1,
                "txt_file_metadata": success2,
            }
        )
    return success1, success2

@observe(name="write_output_to_s3", capture_input=False)
def write_output_to_s3(s3_client, bucket_name, file_name, data, langfuse_client):
    """
    Write data to S3 bucket
    """
    if langfuse_client:
        langfuse_client.update_current_span(
            input={
                "bucket_name": bucket_name,
                "file_name": file_name
            }
        )
    try:
        # json_string = json.dumps(data, ensure_ascii=False)
        response = s3_client.put_object(
            Bucket=bucket_name,
            Key=file_name,
            Body=data,
            ContentType= 'text/plain; charset=utf-8'
        )

        if response['ResponseMetadata']['HTTPStatusCode'] == 200:
            print(f"Successfully uploaded {file_name} to {bucket_name}")
            return True
        else:
            print(f"Failed to upload {file_name} to {bucket_name}")
            return False

    except ClientError as e:
        print(f"Error occurred: {e}")
        return False

@observe(name="process_file", capture_input=False)
def process_file(bucket_name, object_key, region, langfuse_client):
    """Main file processing operation"""
    if langfuse_client:
        langfuse_client.update_current_span(
            input={"bucket": bucket_name, "key": object_key, "region": region}
        )
    chunk_content_list = process_s3_file(bucket_name, object_key, region=region, langfuse_client=langfuse_client)
    
    if not chunk_content_list:
        return {"status": "error", "message": f"Không thể xử lý file {object_key}"}
    
    result_list = []
    
    for i, file_content in enumerate(chunk_content_list):
        logger.info(f"Xử lý chunk {i+1}/{len(chunk_content_list)} của file {object_key}")
        logger.info(f"Độ dài chunk: {len(file_content)} ký tự")
        
        try:
            result = phan_tich_bao_cao_with_retry(
                file_content=file_content,
                region=region,
                langfuse_client=langfuse_client
            )
            logger.info(f"Hoàn thành chunk {i+1}")
            result_list.append(result)
            
        except Exception as chunk_error:
            logger.error(f"Lỗi khi xử lý chunk {i+1}: {chunk_error}")
            continue
    
    if result_list:
        output_filename = '/'.join(object_key.split('/')[1:])
        output_filename = output_filename.replace('.txt', '_curated.txt')
        
        s3_client = boto3.client('s3', region_name=region)
        success1, success2 = split_result_and_write(s3_client, bucket_name, result_list[-1], output_filename, langfuse_client)
        
        if success1 and success2:
            logger.info(f"Hoàn thành file {output_filename} -> 2 files đã được tạo")
            return {
                "status": "success",
                "output_filename": output_filename,
                "chunks_processed": len(result_list)
            }
        else:
            logger.error("Lỗi khi ghi file lên S3")
            return {"status": "error", "message": "Lỗi khi ghi file lên S3"}
    else:
        logger.error(f"Không có chunk nào của file {object_key} được xử lý thành công!")
        return {"status": "error", "message": "Không có chunk nào được xử lý thành công"}


def lambda_handler(event, context):
    langfuse_client = None
    try:
        langfuse_client = get_langfuse_client()
        logger.info("Lambda function started")
        logger.info(f"Event: {json.dumps(event, default=str)}")
        
        for record in event['Records']:
            bucket_name = record['s3']['bucket']['name']
            object_key = record['s3']['object']['key']
            region = 'us-east-1'
            
            if langfuse_client:
                with langfuse_client.start_as_current_span(
                    name="process_BCPT_txt_to_curated",
                    input={"bucket": bucket_name, "key": object_key}
                ):
                    langfuse_client.update_current_trace(
                        metadata={
                            "s3_bucket": bucket_name,
                            "s3_key": object_key,
                            "environment": ENVIRONMENT
                        }
                    )
                    
                    result = process_file(bucket_name, object_key, region, langfuse_client)
                    
                    langfuse_client.update_current_trace(
                        output=result,
                        tags=[ENVIRONMENT, 'process_BCPT_txt_to_curated']
                    )
                    langfuse_client.update_current_span(output=result)
            else:
                result = process_file(bucket_name, object_key, region, langfuse_client)
        
        logger.info("Lambda function completed successfully")
        if langfuse_client:
            langfuse_client.flush()
            logger.info(f"Wait {LANGFUSE_FLUSH_TIME_SLEEP} seconds for Langfuse to flush the spans")
            time.sleep(LANGFUSE_FLUSH_TIME_SLEEP)
        
        return {'statusCode': 200, 'body': json.dumps({'message': 'Processing completed'})}
        
    except Exception as e:
        error_msg = f"Lambda execution failed: {str(e)}"
        logger.error(error_msg)
        
        if langfuse_client:
            langfuse_client.flush()
            logger.info(f"Wait {LANGFUSE_FLUSH_TIME_SLEEP} seconds for Langfuse to flush the spans")
            time.sleep(LANGFUSE_FLUSH_TIME_SLEEP)
        
        return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}