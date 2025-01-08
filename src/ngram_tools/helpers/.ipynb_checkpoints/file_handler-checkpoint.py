import lz4.frame
import orjson


class FileHandler:
    """
    A class to handle file operations, including compression and serialization.
    """

    def __init__(self, path, is_output=False, compress=False):
        """
        Initializes the FileHandler instance.

        Args:
            path (str): The file path to open.
            mode (str): The mode to open the file in.
        """
        self.path = path
        self.compress = compress if is_output else path.endswith('.lz4')
        self.binary_mode = self.compress
        self.mode = (
            'wb' if is_output and self.compress else
            'w' if is_output else
            'rb' if self.compress else
            'r'
        )
        self.encoding = None if self.binary_mode else 'utf-8'
        self.open_fn = lz4.frame.open if self.compress else open

    def open(self):
        """Opens the file in the specified mode."""
        return self.open_fn(self.path, self.mode, encoding=self.encoding)

    def serialize(self, entry):
        """
        Serializes the entry to the appropriate format (compressed or plain).

        Args:
            entry (dict): The entry to serialize.

        Returns:
            str or bytes: Serialized entry in the correct format.
        """
        serialized = orjson.dumps(entry)
        return serialized + b'\n' if self.compress else serialized.decode(
            'utf-8') + '\n'

    def deserialize(self, line):
        """
        Deserializes a line into an entry.

        Args:
            line (str or bytes): The line to deserialize.

        Returns:
            dict: The deserialized entry.
        """
        return orjson.loads(line)