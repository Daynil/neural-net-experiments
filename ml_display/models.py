from typing import Literal

from pydantic import BaseModel
from starlette.authentication import BaseUser
from starlette.requests import Request


class AlertInfo(BaseModel):
    type: Literal["good", "bad", "alert", "info"]
    title: str
    message: str
