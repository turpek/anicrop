import pytest
import numpy as np
from unittest.mock import patch
from anicrop.persistence.manager import ScratchDiskManager
from anicrop.persistence.token import NdarrayToken

# --- Tests for ScratchDiskManager ---


def test_manager_workspace_initialization():
    """Cenário 1: Inicialização do Workspace"""
    manager = ScratchDiskManager()
    try:
        assert manager.workspace_path.exists()
        assert manager.workspace_path.is_dir()
    finally:
        manager.cleanup_session()


def test_manager_save_load_flow():
    """Cenário 2: Fluxo Completo de Save e Load"""
    manager = ScratchDiskManager()
    try:
        array = np.random.rand(100, 100).astype(np.float32)

        file_id = manager.save_array(array)

        assert file_id.endswith(".npy")
        loaded_array = manager.load_array(file_id)

        np.testing.assert_array_equal(array, loaded_array)
    finally:
        manager.cleanup_session()


def test_manager_load_non_existent():
    """Cenário 3: Exceção ao Carregar Array Inexistente"""
    manager = ScratchDiskManager()
    try:
        with pytest.raises(FileNotFoundError, match="já foi limpo ou perdido"):
            manager.load_array("id_falso.npy")
    finally:
        manager.cleanup_session()


def test_manager_deletion():
    """Cenário 4: Deleção Bem-sucedida"""
    manager = ScratchDiskManager()
    try:
        array = np.array([1, 2, 3])
        file_id = manager.save_array(array)

        file_path = manager.workspace_path / file_id
        assert file_path.exists()

        manager.delete_array(file_id)
        assert not file_path.exists()

        with pytest.raises(FileNotFoundError):
            manager.load_array(file_id)
    finally:
        manager.cleanup_session()


def test_manager_idempotent_deletion():
    """Cenário 5: Deleção Silenciosa (Idempotência)"""
    manager = ScratchDiskManager()
    try:
        # Não deve levantar exceção
        manager.delete_array("id_que_nao_existe.npy")
    finally:
        manager.cleanup_session()


# --- Tests for NdarrayToken ---

@patch("anicrop.persistence.token.manager_global")
def test_token_creation_and_save_delegation(mock_manager):
    """Cenário 1: Criação e Delegação de Save"""
    mock_manager.save_array.return_value = "mock_id.npy"
    array_mock = np.array([4, 5, 6])

    token = NdarrayToken(array_mock)

    mock_manager.save_array.assert_called_once_with(array_mock)
    assert token._file_id == "mock_id.npy"


@patch("anicrop.persistence.token.manager_global")
def test_token_restoration(mock_manager):
    """Cenário 2: Restauração do Array"""
    mock_manager.save_array.return_value = "mock_id.npy"
    expected_array = np.array([7, 8, 9])
    mock_manager.load_array.return_value = expected_array

    token = NdarrayToken(np.zeros(3))
    restored_array = token.restore()

    mock_manager.load_array.assert_called_once_with("mock_id.npy")
    np.testing.assert_array_equal(restored_array, expected_array)


@patch("anicrop.persistence.token.manager_global")
def test_token_destruction(mock_manager):
    """Cenário 3: Destruição do Token"""
    mock_manager.save_array.return_value = "mock_id.npy"
    token = NdarrayToken(np.zeros(3))

    token.destroy()

    mock_manager.delete_array.assert_called_once_with("mock_id.npy")
    assert token._file_id is None


@patch("anicrop.persistence.token.manager_global")
def test_token_idempotent_destruction(mock_manager):
    """Cenário 4: Destruição Idempotente (Dupla chamada)"""
    mock_manager.save_array.return_value = "mock_id.npy"
    token = NdarrayToken(np.zeros(3))

    token.destroy()
    token.destroy()

    # Deve ser chamado apenas uma vez devido ao check 'if self._file_id'
    assert mock_manager.delete_array.call_count == 1
