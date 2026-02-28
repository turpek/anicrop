from anicrop.persistence.manager import manager_global
import numpy as np


class NdarrayToken:
    """Referência leve que substitui a imagem pesada na RAM."""

    def __init__(self, array: np.ndarray):
        self._manager = manager_global
        self._file_id = self._manager.save_array(array)

    def restore(self) -> np.ndarray:
        """Puxa o array de volta do disco."""
        return self._manager.load_array(self._file_id)

    def destroy(self):
        """Avisa o disco para apagar o arquivo permanentemente."""
        if self._file_id:
            self._manager.delete_array(self._file_id)
            self._file_id = None
