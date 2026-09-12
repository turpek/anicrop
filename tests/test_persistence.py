import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

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


def test_manager_stale_cleanup_removes_dead_process_directories(tmp_path: Path):
    """Valida que diretorios temporarios de processos finalizados sao removidos na inicializacao."""
    base_dir = tmp_path / "temp_shm"
    base_dir.mkdir()
    dead_dir = base_dir / "anicrop_scratch_9999999_dead_uuid"
    dead_dir.mkdir()
    (dead_dir / "test.raw").write_bytes(b"dummy")

    manager = ScratchDiskManager(base_dir=base_dir)
    assert not dead_dir.exists()
    manager.cleanup_session()


def test_manager_stale_cleanup_preserves_alive_process_directories(tmp_path: Path):
    """Valida que diretorios temporarios de processos ativos sao preservados na inicializacao."""
    base_dir = tmp_path / "temp_shm"
    base_dir.mkdir()
    alive_dir = base_dir / f"anicrop_scratch_{os.getpid()}_alive_uuid"
    alive_dir.mkdir()
    (alive_dir / "test.raw").write_bytes(b"active")

    manager = ScratchDiskManager(base_dir=base_dir)
    assert alive_dir.exists()
    manager.cleanup_session()
    shutil.rmtree(alive_dir, ignore_errors=True)


def test_manager_check_disk_space_raises_on_insufficient_space(tmp_path: Path):
    """Valida que check_disk_space lanca OSError quando o espaco livre e insuficiente."""
    manager = ScratchDiskManager(base_dir=tmp_path)
    mock_usage = MagicMock(free=100 * 1024 * 1024)

    with patch("shutil.disk_usage", return_value=mock_usage):
        with pytest.raises(OSError, match="Espaço insuficiente em disco"):
            manager.check_disk_space(
                tmp_path,
                required_bytes=10 * 1024 * 1024,
                headroom_bytes=128 * 1024 * 1024,
            )
    manager.cleanup_session()


def test_manager_get_temp_file_path_tiered_fallback(tmp_path: Path):
    """Valida que get_temp_file_path recorre ao disco secundario quando a memoria compartilhada e pequena."""
    primary_dir = tmp_path / "primary_shm"
    primary_dir.mkdir()
    manager = ScratchDiskManager(base_dir=primary_dir)

    workspace_str = str(manager.workspace_path)
    lookup = {
        workspace_str: MagicMock(free=50 * 1024 * 1024),
    }
    fallback_usage = MagicMock(free=2 * 1024 * 1024 * 1024)

    with patch("shutil.disk_usage", side_effect=lambda p: lookup.get(str(p), fallback_usage)):
        temp_file = manager.get_temp_file_path(required_bytes=200 * 1024 * 1024)

    assert "anicrop_scratch_disk_" in str(temp_file.parent)
    assert manager._disk_fallback_dir is not None
    manager.cleanup_session()
    assert manager._disk_fallback_dir is None


def test_manager_get_temp_file_path_raises_when_all_targets_exhausted(tmp_path: Path):
    """Valida que get_temp_file_path lanca OSError preventivo quando nenhum dispositivo tem espaco."""
    manager = ScratchDiskManager(base_dir=tmp_path)
    mock_usage = MagicMock(free=10 * 1024 * 1024)

    with patch("shutil.disk_usage", return_value=mock_usage):
        with pytest.raises(OSError, match="Espaço insuficiente em disco"):
            manager.get_temp_file_path(required_bytes=500 * 1024 * 1024)
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
