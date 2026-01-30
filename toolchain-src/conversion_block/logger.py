from typing import List, Optional, Dict, Any
import datetime
import json
import os

__all__ = ['logs']


class Message:
    def __init__(self):
        self.message: str = ''
        self.data: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        # Emit a consistent schema: always include Message, include Data only when present.
        log: Dict[str, Any] = {'Message': self.message}
        if self.data:
            log['Data'] = self.data
        return log

    def __str__(self) -> str:
        # JSON string representation of a single message.
        return json.dumps(self.to_dict(), default=str)


class Logs:
    def __init__(self):
        self.messages: List[Message] = []

    def add_message(self, message: str, data: Optional[dict] = None) -> None:
        # Append a new message entry and print for immediate visibility in CLI runs.
        msg = Message()
        msg.message = message
        msg.data = data or {}
        self.messages.append(msg)
        print(f'{message}: {msg.data}')

    def add_data(self, **data: Any) -> None:
        # Attach additional key/value fields to the most recent message.
        # If no message exists yet, create a placeholder message.
        try:
            self.messages[-1].data.update(data)
        except IndexError:
            self.add_message('<Empty Message>', data)
        print(data)

    def save_as_json(self, path: Optional[str] = None) -> str:
        # Persist logs as pretty-printed JSON.
        # If no path is provided, write under /tmp with a filesystem-safe timestamp.
        if path is None:
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            path = os.path.join('/', 'tmp', f'{ts}.json')

        # Ensure the parent directory exists.
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        # Write using UTF-8 for consistent behavior across environments.
        with open(path, 'w', encoding='utf-8') as f:
            f.write(str(self))

        return path

    def __str__(self) -> str:
        # Serialize all messages as a JSON array.
        messages_as_dict = [msg.to_dict() for msg in self.messages]
        return json.dumps(messages_as_dict, default=str, indent=2)


logs = Logs()
