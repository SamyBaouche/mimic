"""Read emails as tasks: a folder of .eml files (what Outlook, Gmail, Thunderbird export).

Each email becomes one item of a workflow, with its attachments saved where the agent can
read them and upload them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from email import policy
from email.parser import BytesParser
from email.utils import parseaddr
from pathlib import Path


@dataclass
class Email:
    id: str
    sender: str  # "Bureau Express Billing <billing@...>"
    address: str  # billing@...
    name: str
    subject: str
    date: str
    body: str
    attachments: list[Path] = field(default_factory=list)

    def fields(self) -> dict:
        """What a workflow can use: {email.subject}, {email.address}, {email.attachment}..."""
        return {
            "id": self.id,
            "sender": self.sender,
            "address": self.address,
            "name": self.name,
            "subject": self.subject,
            "date": self.date,
            "body": self.body,
            "attachment": str(self.attachments[0]) if self.attachments else "",
        }


def read_email(path: Path, attachments_dir: Path) -> Email:
    msg = BytesParser(policy=policy.default).parse(path.open("rb"))
    name, address = parseaddr(str(msg["From"] or ""))
    body = msg.get_body(preferencelist=("plain", "html"))
    saved = []
    for part in msg.iter_attachments():
        filename = Path(part.get_filename() or "attachment.bin").name  # never a path from mail
        target = attachments_dir / path.stem / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(part.get_payload(decode=True) or b"")
        saved.append(target)
    return Email(
        id=path.stem,
        sender=str(msg["From"] or ""),
        address=address,
        name=name or address,
        subject=str(msg["Subject"] or ""),
        date=str(msg["Date"] or ""),
        body=body.get_content().strip() if body else "",
        attachments=saved,
    )


def read_mailbox(folder: Path, attachments_dir: Path) -> list[Email]:
    files = sorted(Path(folder).glob("*.eml"))
    if not files:
        raise FileNotFoundError(f"No .eml files in {folder}")
    return [read_email(p, attachments_dir) for p in files]
