import socket
import struct
import time

from battle_model.play import RecordServer


def test_a_record_larger_than_the_first_buffer_arrives_whole():
	"""play.RecordServer: a battle's record of several MB (its buffer doubles twice) arrives as sent, its key cut at #
	(2026-11-05: a view of the buffer outliving its recv_into under Python 3.14 made the doubling raise BufferError and
	the record was lost)."""
	payload = bytes(range(256)) * (3 * 4096 + 7)  # (3 MB and a bit: past 1 MB and 2 MB)
	key = b"g0:RED:battle#3"
	records = {}
	with RecordServer(concurrent=True) as server:
		with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
			client.connect(str(server.path))
			client.sendall(struct.pack("<I", len(key)) + key + payload)
		for _ in range(250):
			records = server.drain()
			if records:
				break
			time.sleep(0.02)
	assert list(records) == ["g0:RED:battle"] and records["g0:RED:battle"] == [payload]
