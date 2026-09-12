import ctypes
import os
import shutil
import tempfile
import uuid
from pathlib import Path

import numpy as np

DEFAULT_MIN_DISK_HEADROOM: int = 128 * 1024 * 1024  # 128 MB


def _resolve_optimal_temp_dir() -> str | None:
    """Detecta se /dev/shm está disponível no Linux com espaço livre suficiente (>512MB)."""
    shm_path = Path("/dev/shm")
    if shm_path.exists() and shm_path.is_dir() and os.access(shm_path, os.W_OK):
        try:
            usage = shutil.disk_usage(shm_path)
            if usage.free >= 512 * 1024 * 1024:
                return str(shm_path)
        except OSError:
            pass
    return None


def _format_bytes(num_bytes: float | int) -> str:
    """Formata bytes em representação legível (MB ou GB)."""
    if num_bytes >= 1024**3:
        return f"{num_bytes / (1024**3):.2f} GB"
    return f"{num_bytes / (1024**2):.1f} MB"


def _is_pid_alive(pid: int) -> bool:
    """Verifica se um processo com o PID informado está em execução (cross-platform)."""
    if pid <= 0:
        return False

    if os.name == "nt":
        windll = getattr(ctypes, "windll", None)
        if windll is not None:
            process_query_limited_info = 0x1000
            synchronize = 0x00100000
            handle = windll.kernel32.OpenProcess(
                process_query_limited_info | synchronize, False, pid
            )
            if handle:
                windll.kernel32.CloseHandle(handle)
                return True
            return False
        return True

    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _cleanup_stale_workspaces(base_dir: str | Path | None) -> None:
    """Remove pastas temporárias anicrop_scratch_* de processos que já morreram."""
    if base_dir is None:
        return
    base_path = Path(base_dir)
    if not base_path.exists() or not base_path.is_dir():
        return
    try:
        for item in base_path.glob("anicrop_scratch_*"):
            if not item.is_dir():
                continue
            parts = item.name.split("_")
            if len(parts) >= 3 and parts[2].isdigit():
                owner_pid = int(parts[2])
                if not _is_pid_alive(owner_pid):
                    shutil.rmtree(item, ignore_errors=True)
            elif len(parts) >= 4 and parts[3].isdigit():
                owner_pid = int(parts[3])
                if not _is_pid_alive(owner_pid):
                    shutil.rmtree(item, ignore_errors=True)
    except Exception:
        pass


class ScratchDiskManager:
    """Singleton ou instância global por Canvas para gerenciar os arquivos temporários."""

    def __init__(self, base_dir: str | Path | None = None) -> None:
        resolved_base = (
            str(base_dir) if base_dir is not None else _resolve_optimal_temp_dir()
        )
        _cleanup_stale_workspaces(resolved_base)
        pid = os.getpid()
        self._temp_dir = tempfile.TemporaryDirectory(
            prefix=f"anicrop_scratch_{pid}_", dir=resolved_base
        )
        self.workspace_path = Path(self._temp_dir.name)
        self._disk_fallback_dir: tempfile.TemporaryDirectory[str] | None = None
        self.min_disk_headroom: int = DEFAULT_MIN_DISK_HEADROOM

    def check_disk_space(
        self,
        target_dir: Path,
        required_bytes: int,
        headroom_bytes: int | None = None,
    ) -> None:
        """Valida se há espaço livre suficiente no diretório antes de alocar um buffer."""
        if required_bytes <= 0:
            return
        headroom = (
            self.min_disk_headroom if headroom_bytes is None else headroom_bytes
        )
        try:
            usage = shutil.disk_usage(target_dir)
        except OSError:
            return

        available_bytes = usage.free
        total_needed = required_bytes + headroom
        if total_needed > available_bytes:
            req_str = _format_bytes(required_bytes)
            free_str = _format_bytes(available_bytes)
            headroom_str = _format_bytes(headroom)
            raise OSError(
                f"Espaço insuficiente em disco: a operação requer {req_str} "
                f"(com margem de segurança de {headroom_str}), mas restam apenas "
                f"{free_str} disponíveis em '{target_dir}'."
            )

    def get_temp_file_path(self, required_bytes: int = 0) -> Path:
        """Gera um caminho para arquivo temporário no workspace, validando espaço prévio."""
        headroom = self.min_disk_headroom
        total_needed = required_bytes + headroom

        # 1. Tenta usar o workspace_path primário (frequentemente /dev/shm)
        primary_free: float = float("inf")
        try:
            primary_free = float(shutil.disk_usage(self.workspace_path).free)
        except OSError:
            pass

        if required_bytes <= 0 or total_needed <= primary_free:
            return self.workspace_path / f"mmap_{uuid.uuid4().hex}.raw"

        # 2. Se o primário não tiver espaço suficiente (ex: /dev/shm pequeno), tenta o disco temporário padrão
        disk_temp = Path(tempfile.gettempdir())
        disk_free: float = 0.0
        try:
            disk_free = float(shutil.disk_usage(disk_temp).free)
        except OSError:
            pass

        if total_needed <= disk_free:
            if self._disk_fallback_dir is None:
                _cleanup_stale_workspaces(disk_temp)
                pid = os.getpid()
                self._disk_fallback_dir = tempfile.TemporaryDirectory(
                    prefix=f"anicrop_scratch_disk_{pid}_", dir=str(disk_temp)
                )
            return Path(self._disk_fallback_dir.name) / f"mmap_{uuid.uuid4().hex}.raw"

        # 3. Nem o primário nem o disco suportam a alocação: rejeita preventivamente
        best_dir = self.workspace_path if primary_free >= disk_free else disk_temp
        best_free = max(primary_free, disk_free)
        req_str = _format_bytes(required_bytes)
        free_str = _format_bytes(best_free)
        headroom_str = _format_bytes(headroom)
        raise OSError(
            f"Espaço insuficiente em disco: a operação requer {req_str} "
            f"(com margem de segurança de {headroom_str}), mas restam apenas "
            f"{free_str} disponíveis em '{best_dir}'."
        )

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
        if self._disk_fallback_dir is not None:
            self._disk_fallback_dir.cleanup()
            self._disk_fallback_dir = None
        self._temp_dir.cleanup()


manager_global = ScratchDiskManager()
