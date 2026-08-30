import cv2
import numpy as np

from anicrop.document import Document
from anicrop.type import Scale
from anicrop.viewport import Viewport


class Viewer:
    """
    Uma janela de visualização embutida usando OpenCV para renderizar de forma interativa
    o conteúdo de um Documento através de uma Viewport.
    """

    def __init__(self, doc: Document, viewport: Viewport):
        self.doc = doc
        self.viewport = viewport

        # O título da janela usa o nome do documento
        self.window_name = f"Anicrop: {self.doc.name}"

        # Trava a janela para não ser redimensionada pelo usuário
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

    def fit_canvas(self) -> None:
        """
        Calcula a escala exata para que o Canvas do Documento caiba inteiramente
        dentro da Viewport e ajusta o zoom da câmera.
        """
        cw, ch = self.doc.canvas.size
        vw, vh = self.viewport.size

        # Qual a escala máxima que podemos aplicar sem estourar as bordas?
        fit_scale = min(vw / cw, vh / ch)

        # Aplica a escala na viewport. O método interno de fit já vai centralizar tudo.
        self.viewport._fit = Scale(fit_scale, fit_scale)

    def show(self) -> None:
        """
        Gera a renderização da cena e exibe (ou atualiza) na janela.
        A janela terá a dimensão exata definida na Viewport.
        """
        img = self.doc.preview(self.viewport)

        if img is None:
            # Mostra uma tela vazia se nada renderizou
            vw, vh = self.viewport.size
            frame_display = np.zeros((vh, vw, 3), dtype=np.uint8)
            cv2.imshow(self.window_name, frame_display)
            return

        cv2.imshow(self.window_name, img.bgr())

    def wait(self, delay: int = 0) -> int:
        """
        Aguarda uma tecla ser pressionada.
        Retorna o código ASCII da tecla pressionada.
        """
        return cv2.waitKey(delay) & 0xFF

    def close(self) -> None:
        """Destrói a janela de visualização."""
        cv2.destroyWindow(self.window_name)

    def __del__(self) -> None:
        """Garante que a janela seja destruída quando o objeto for coletado pelo Garbage Collector."""
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass
