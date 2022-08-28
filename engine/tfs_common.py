from dataclasses import dataclass


@dataclass
class TFSResult:
    data: list = None       
    err_msg: str = ""