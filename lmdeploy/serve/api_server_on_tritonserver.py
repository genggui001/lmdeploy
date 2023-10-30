# Copyright (c) genggui001. All rights reserved.
import os
import json
import time
import random
from typing import Optional, List, Literal, Union
from pydantic import BaseModel, Field
from lmdeploy.serve.turbomind.chatbot import Chatbot, StatusCode, get_logger

from flask import Flask, Response, stream_with_context
from flask_pydantic import validate
from werkzeug.exceptions import HTTPException, BadRequest, InternalServerError

app = Flask(__name__)

log_level = os.environ.get('SERVICE_LOG_LEVEL', 'INFO')
TRITONSERVER_ADDR = os.environ.get('TRITONSERVER_ADDR')
SESSION_LEN = int(os.environ.get('SESSION_LEN', "-1"))
ADD_EOS = (os.environ.get('ADD_EOS', "1") == '1')

assert SESSION_LEN > 0

logger = get_logger(log_level=log_level)

class CompletionRequest(BaseModel):
    """Chat completion request."""
    model: str
    prompt: str
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = 50
    n: Optional[int] = 1
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False
    repetition_penalty: Optional[float] = 1.0

class UsageInfo(BaseModel):
    """Usage information."""
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0

class CompletionResponseChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[float] = None
    finish_reason: Optional[Literal['stop', 'length']] = None


class CompletionResponse(BaseModel):
    """Chat completion response."""
    id: str
    object: str = 'completion'
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo


class CompletionStreamResponse(BaseModel):
    """Chat completion stream response."""
    id: str
    object: str = 'completion.chunk'
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: List[CompletionResponseChoice]
    model: str


app.config['TRAP_HTTP_EXCEPTIONS'] = True
@app.errorhandler(HTTPException)
def handle_exception(e: HTTPException):
    """Return JSON instead of HTML for HTTP errors."""
    # start with the correct headers and status code from the error
    response = e.get_response()
    # replace the body with JSON
    response.data = json.dumps({
        "error": {
            "message": e.description,
            "type": e.name,
            "code": e.code,
            "param": None,
        }
    }, ensure_ascii=False)
    response.content_type = "application/json"
    return response


@app.post('/v1/completions')
@validate()
def completions_v1(
    body: CompletionRequest,
):
    assert body.n == 1

    chatbot = Chatbot(
        TRITONSERVER_ADDR,
        log_level=log_level,
        display=False,
        temperature=body.temperature,
        top_p=body.top_p,
        top_k=body.top_k,
        repetition_penalty=body.repetition_penalty,
        session_len=SESSION_LEN,
        add_eos=ADD_EOS,
    )
    session_id = random.randint(100000, 9999999)

    logger.info("CompletionRequest: " + json.dumps(dict(
        session_id=session_id,
        **body.dict()
    ), ensure_ascii=False, indent=4))

    def generate(redata_index):
        all_response_text = ""
        
        # sequence_start clean cache
        for status, res, tokens in chatbot.stream_infer(
            session_id=session_id,
            prompt=body.prompt,
            request_output_len=body.max_tokens,
            sequence_start=True,
        ):
            
            if status == StatusCode.TRITON_SESSION_OUT_OF_LIMIT:
                raise BadRequest(description="request length out of limit")
            
            if status == StatusCode.TRITON_SERVER_ERR:
                raise InternalServerError(description=str(res))
            
            finish_reason = None
            if status == StatusCode.TRITON_STREAM_END:
                finish_reason = "stop"

            yield CompletionResponseChoice(
                text=res[len(all_response_text):],
                index=redata_index,
                finish_reason=finish_reason,
            )

            all_response_text = res


    if body.stream == True:

        # stream
        def stream_generator(context_iter, choice):
            try:
                while True:
                    response = CompletionStreamResponse(
                        id=str(session_id),
                        choices=[choice],
                        model=body.model
                    )
                    yield f'data: {response.json(ensure_ascii=False)}\n\n'.encode("utf8")

                    try:
                        choice = next(context_iter)
                    except StopIteration:
                        # out
                        break

                yield 'data: [DONE]\n\n'

            except GeneratorExit:
                # cancel
                chatbot.cancel(session_id=session_id)
                logger.info(f"session_id = {session_id} stop") 
            
            # 关闭迭代器
            context_iter.close()
            del choice
        
        context_iter = generate(redata_index=0)
        try:
            first_item = next(context_iter)
        except Exception as e:
            # 关闭迭代器
            context_iter.close()
            raise e

        return Response(response=stream_with_context(stream_generator(context_iter, first_item)), status=200)

    else:
        usage_info = UsageInfo(
            prompt_tokens=0,
            total_tokens=0,
            completion_tokens=0,
        )
        final_choice = CompletionResponseChoice(
            text="",
            index=0,
            finish_reason=None,
        )

        for choice in generate(redata_index=0):
            final_choice.text += choice.text
            final_choice.finish_reason = choice.finish_reason

            usage_info.completion_tokens += 1
            usage_info.total_tokens += 1

        return CompletionResponse(
            id=str(session_id),
            choices=[final_choice],
            model=body.model,
            usage=usage_info,
        ).json(ensure_ascii=False).encode("utf8"), {"Content-Type": "application/json"}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10045)
