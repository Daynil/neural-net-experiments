from dataclasses import asdict, dataclass
from queue import Queue
from typing import Literal, Optional, Union


@dataclass
class LearnerStats:
    run_id: str
    total_epochs: int
    current_epoch: int
    current_item: int
    total_items: int
    num_batches: int
    loss_value: float
    seconds_per_item: float
    loop_type: Literal["train", "valid", "test"]
    accuracy: Optional[float] = -0.1

    def as_dict(self):
        return asdict(self)


# TODO: consider what types queue can have and make union
Message = Union[str, LearnerStats]
MessageQueue = Queue[Message]
message_queue = Queue[Message]()

ClientRequest = Literal['cancel', 'placeholder']
ClientRequestQueue = Queue[ClientRequest]
client_request_queue = Queue[ClientRequest]()
