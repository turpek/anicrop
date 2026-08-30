import tempfile
import uuid
from pathlib import Path

import numpy as np


class ScratchDiskManager:
    """Singleton ou instância global por Canvas para gerenciar os arquivos temporários."""

    def __init__(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory(prefix="anicrop_scratch_")
        self.workspace_path = Path(self._temp_dir.name)

    def save_array(self, array: np.ndarray) -> str:
        file_id = f"{uuid.uuid4().hex}.npy"
        file_path = self.workspace_path / file_id

        np.save(file_path, array)
        return file_id

    def load_array(self, file_id: str) -> np.ndarray:
        file_path = self.workspace_path / file_id
        if not file_path.exists():
            raise FileNotFoundError(f"Array {file_id} já foi limpo ou perdido.")

        return np.load(file_path)

    def delete_array(self, file_id: str) -> None:
        file_path = self.workspace_path / file_id
        if file_path.exists():
            file_path.unlink()

    def cleanup_session(self) -> None:
        self._temp_dir.cleanup()


manager_global = ScratchDiskManager()
