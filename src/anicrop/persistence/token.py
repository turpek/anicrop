import numpy as np

from anicrop.persistence.manager import manager_global


class NdarrayToken:
    """Referência leve que substitui a imagem pesada na RAM."""

    def __init__(self, array: np.ndarray) -> None:
        self._manager = manager_global
        self._file_id: str | None = self._manager.save_array(array)

    def restore(self) -> np.ndarray:
        """Puxa o array de volta do disco."""
        if self._file_id is None:
            raise RuntimeError("Token já foi destruído ou não possui arquivo associado.")
        return self._manager.load_array(self._file_id)

    def destroy(self) -> None:
        """Avisa o disco para apagar o arquivo permanentemente."""
        if self._file_id is not None:
            self._manager.delete_array(self._file_id)
            self._file_id = None
