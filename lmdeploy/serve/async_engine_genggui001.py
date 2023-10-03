# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import dataclasses
import os.path as osp
import random
from contextlib import contextmanager
from typing import Literal, Optional

from lmdeploy.model import MODELS, BaseModel


@dataclasses.dataclass
class GenOut:
    """Pack all response information together."""
    instance_id: int
    response: str
    history_token_len: int
    input_token_len: int
    generate_token_len: int
    finish_reason: Optional[Literal['stop', 'length']] = None


class AsyncEngine:
    """Async inference engine. Maintaining a bunch of tm_model instances.

    Args:
        model_path (str): the path of the deployed model
        instance_num (int): instance numbers to be created
        tp (int): tensor parallel
    """

    def __init__(self, model_path, instance_num=32, tp=1) -> None:
        from lmdeploy import turbomind as tm
        from lmdeploy.turbomind.tokenizer import Tokenizer
        tokenizer_model_path = osp.join(model_path, 'triton_models',
                                        'tokenizer')
        tokenizer = Tokenizer(tokenizer_model_path)
        self.tm_model = tm.TurboMind(model_path,
                                     eos_id=tokenizer.eos_token_id,
                                     tp=tp)
        self.tokenizer = tokenizer
        self.generators = [
            self.tm_model.create_async_instance() for i in range(instance_num)
        ]
        self.instance_num = instance_num
        self.model: BaseModel = MODELS.get(self.tm_model.model_name)()
        self.available = [asyncio.Lock() for _ in range(instance_num)]


    async def stop_generate_openai(
        self,
        instance_id,
    ):
        """Generate responses.

        Args:
            messages (str | List): chat history or prompt
            instance_id (int): actually request host ip
            stream_response (bool): whether return responses streamingly
            renew_session (bool): renew the session
            request_output_len (int): output token nums
            stop (bool): whether stop inference
            top_p (float): If set to float < 1, only the smallest set of most
              probable tokens with probabilities that add up to top_p or higher
            are kept for generation.
            top_k (int): The number of the highest probability vocabulary
              tokens to keep for top-k-filtering
            temperature (float): to modulate the next token probability
            repetition_penalty (float): The parameter for repetition penalty.
              1.0 means no penalty
            ignore_eos (bool): indicator for ignoring eos
        """
        session_id = instance_id
        instance_id %= self.instance_num

        generator = self.generators[instance_id]

        async for _ in generator.async_stream_infer(
            session_id=session_id,
            input_ids=[self.tokenizer.encode('')],
            stream_output=True,
            request_output_len=0,
            sequence_start=False,
            sequence_end=False,
            step=0,
            stop=True,
        ):

            yield GenOut(
                instance_id=session_id,
                response="", 
                history_token_len=0,    
                input_token_len=0, 
                generate_token_len=0, 
                finish_reason='stop'
            )


    async def generate_openai(
        self,
        messages,
        instance_id,
        request_output_len=512,
        top_k=40,
        top_p=0.8,
        temperature=0.8,
        repetition_penalty=1.0,
        ignore_eos=False,
    ):
        """Generate responses.

        Args:
            messages (str | List): chat history or prompt
            instance_id (int): actually request host ip
            stream_response (bool): whether return responses streamingly
            renew_session (bool): renew the session
            request_output_len (int): output token nums
            stop (bool): whether stop inference
            top_p (float): If set to float < 1, only the smallest set of most
              probable tokens with probabilities that add up to top_p or higher
            are kept for generation.
            top_k (int): The number of the highest probability vocabulary
              tokens to keep for top-k-filtering
            temperature (float): to modulate the next token probability
            repetition_penalty (float): The parameter for repetition penalty.
              1.0 means no penalty
            ignore_eos (bool): indicator for ignoring eos
        """
        session_id = instance_id
        instance_id %= self.instance_num
        sequence_start = True

        prompt = self.model.messages2prompt(messages, sequence_start)
        input_ids = self.tokenizer.encode(prompt)
        
        if len(input_ids) >= self.tm_model.session_len:
            yield GenOut(
                instance_id=session_id,
                response='', 
                history_token_len=0, 
                input_token_len=len(input_ids), 
                generate_token_len=0,  
                finish_reason='length'
            )

            return

        # start
        async with self.available[instance_id]:
            generator = self.generators[instance_id]

            # renew a session
            async for _ in generator.async_stream_infer(
                session_id=session_id,
                input_ids=[self.tokenizer.encode('')],
                request_output_len=0,
                sequence_start=False,
                sequence_end=True
            ):
                pass

            finish_reason = None
            seed = random.getrandbits(64)
            response_size = 0
            async for outputs in generator.async_stream_infer(
                session_id=session_id,
                input_ids=[input_ids],
                stream_output=True,
                request_output_len=request_output_len,
                sequence_start=(sequence_start),
                sequence_end=False,
                step=0,
                stop=False,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                ignore_eos=ignore_eos,
                random_seed=seed if sequence_start else None
            ):
                res, tokens = outputs[0]
                # decode res removw chr(65533) == '�'
                all_response = self.tokenizer.decode(res.tolist(), offset=0).strip(chr(65533))
                response = all_response[response_size:]
                response_size = len(response_size)

                # response, history token len, input token len, gen token len
                if len(response) > 0:
                    yield GenOut(
                        instance_id=session_id,
                        response=response, 
                        history_token_len=0,    
                        input_token_len=len(input_ids), 
                        generate_token_len=tokens, 
                        finish_reason=finish_reason
                    )


