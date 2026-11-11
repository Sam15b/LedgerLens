Bedrock `Converse` / `ConverseStream` responses used by
`tests/unit/llm/providers/test_bedrock_provider.py` and
`tests/unit/llm/test_structured_contract.py`.

They are boto3's parsed responses: `client.converse(...)` returns a dict, and the events of
`client.converse_stream(...)["stream"]` are listed in order. The structured reply starts
with a reasoning block, as Claude models with adaptive thinking return. Ids are
illustrative. Re-record with AWS credentials, e.g.:

    python -c "import boto3, json; print(json.dumps(boto3.client('bedrock-runtime').converse(
        modelId='global.anthropic.claude-sonnet-5-5',
        messages=[{'role': 'user', 'content': [{'text': 'What is a PO? One sentence.'}]}]),
        default=str))"
