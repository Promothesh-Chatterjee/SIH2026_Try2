"""Bridge from RadioEnvironment events to the SieveReceiver."""

from __future__ import annotations

from typing import List

from .sieve_receiver import SieveReceiver

__all__ = ["RadioReceiverBridge", "attach_receiver"]


class RadioReceiverBridge:
    def __init__(self, receiver: SieveReceiver) -> None:
        self.receiver = receiver

    def on_event(self, event) -> None:
        if event is None:
            return
        self.receiver.handle_environment_event(event)

    def scan_steps(self, n: int = 1) -> List:
        count = int(n)
        if count < 0:
            raise ValueError("n must be >= 0")
        return self.receiver.scan(count)


def attach_receiver(env, receiver: SieveReceiver) -> RadioReceiverBridge:
    bridge = RadioReceiverBridge(receiver)
    env.add_callback(bridge.on_event)
    return bridge
