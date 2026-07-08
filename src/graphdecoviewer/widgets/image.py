import io
import numpy as np
from typing import Optional
from PIL import Image as _PILImage
from . import Widget
from OpenGL.GL import *
from abc import abstractmethod
from ..types import *

# PIL image modes for encode/decode, keyed by channel count. JPEG has no
# alpha channel, so 4-channel images can only use "png".
_PIL_MODE_BY_CHANNELS = {1: "L", 3: "RGB", 4: "RGBA"}


def _encode_image(img: np.ndarray, compression: str, quality: int) -> bytes:
    """Compress an (H, W, C) or (H, W) uint8 array to `compression`-format bytes."""
    if img.ndim == 2:
        img = img[..., None]
    channels = img.shape[-1]
    mode = _PIL_MODE_BY_CHANNELS.get(channels)
    if mode is None:
        raise ValueError(f"Can't compress an image with {channels} channels.")
    if compression == "jpeg" and mode == "RGBA":
        raise ValueError("JPEG doesn't support an alpha channel; use compression='png'.")
    arr = np.ascontiguousarray(img)
    if channels == 1:
        arr = arr[..., 0]
    pil_img = _PILImage.fromarray(arr, mode=mode)
    buf = io.BytesIO()
    if compression == "jpeg":
        pil_img.save(buf, format="JPEG", quality=quality)
    elif compression == "png":
        pil_img.save(buf, format="PNG")
    else:
        raise ValueError(f"Unknown compression format: {compression!r} (expected 'jpeg' or 'png')")
    return buf.getvalue()


def _decode_image(binary: bytes) -> np.ndarray:
    """Decompress bytes produced by `_encode_image` back to an (H, W, C) uint8 array."""
    pil_img = _PILImage.open(io.BytesIO(binary))
    arr = np.asarray(pil_img)
    return arr[..., None] if arr.ndim == 2 else arr


def _cudaGetErrorEnum(error):
    if isinstance(error, driver.CUresult):
        err, name = driver.cuGetErrorName(error)
        return name if err == driver.CUresult.CUDA_SUCCESS else "<unknown>"
    else:
        raise RuntimeError('Unknown error type: {}'.format(error))

def checkCudaErrors(result):
    if result[0].value:
        raise RuntimeError("CUDA error code={}({})".format(result[0].value, _cudaGetErrorEnum(result[0])))
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]

class Image(Widget):
    """
    Base class for the image viewer widget. Each child class must override
    the '_upload' method to upload their image to the OpenGL texture.
    """
    def __init__(self, mode: ViewerMode, compression: Optional[str] = None, quality: int = 85):
        """
        `compression`: None (default) streams raw uint8 bytes over `server_send`/
        `client_recv` -- exact fidelity, but bandwidth-hungry (e.g. 1280x720 RGB
        is ~2.7 MB *per frame*). Set to "jpeg" (lossy, smaller, needs 3-channel
        RGB) or "png" (lossless, supports 1/3/4 channels) to compress each frame
        before sending -- worth it whenever the link's bandwidth, not render
        time, is the bottleneck (e.g. a slow or SSH-tunneled remote connection).
        `quality` (0-100) only applies to "jpeg".
        """
        self._compression = compression
        self._quality = quality
        self.texture = Texture2D()
        self.img = None
        self.step_called = False
        super().__init__(mode)

    def setup(self):
        """ Create OpenGL texture to be displayed. """
        if self.mode & LOCAL_CLIENT:
            self.texture.id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.texture.id)
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    
    def destroy(self):
        """ Delete the texture. """
        if self.mode & LOCAL_CLIENT:
            glDeleteTextures(1, int(self.texture.id))

    def step(self, img):
        """
        This function just stores the references to the image to be displayed.
        The actual uploading of the image to the OpenGL texture is done in the
        the 'show_gui' function. If your application changes underlying memory
        of the image, then make sure to copy the image before passing it.
        """
        self.img = img
        self.step_called = True

    @abstractmethod
    def _upload(self):
        """
        Upload 'self.img' to the OpenGL texture. Each child class should override
        this method to define the upload procedure based upon the source.
        """

    def show_gui(self, draw_list: 'imgui.ImDrawList'=None, res_x=0, res_y=0):
        if self.img is None:
            return

        glBindTexture(GL_TEXTURE_2D, self.texture.id)
        self._upload()

        if res_x <= 0:
            res_x = self.texture.res_x
        if res_y <= 0:
            res_y = self.texture.res_y

        if draw_list is not None:
            # Figure out
            draw_list.add_image(self.texture.tex_ref, (0, 0), (res_x, res_y))
        else:
            imgui.image(self.texture.tex_ref, (res_x, res_y))

    def import_client_modules(self):
        global imgui
        from imgui_bundle import imgui

class NumpyImage(Image):
    """ Image viewer where the image to be shown comes from NumPy array. """
    def _upload(self):
        img = self.img
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        if img.shape[1] != self.texture.res_x or img.shape[0] != self.texture.res_y:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, img.shape[1], img.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, img)
            self.texture.res_x = img.shape[1]
            self.texture.res_y = img.shape[0]
        else:
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, img.shape[1], img.shape[0], GL_RGB, GL_UNSIGNED_BYTE, img)
    
    def server_send(self):
        if not self.step_called:
            return None, None
        img = self.img
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        self.step_called = False
        if self._compression is not None:
            binary = _encode_image(img, self._compression, self._quality)
            return binary, {"format": self._compression}
        return memoryview(np.ascontiguousarray(img.flatten())), {"shape": tuple(img.shape), "format": "raw"}

    def client_recv(self, binary, text):
        if text.get("format", "raw") == "raw":
            self.img = np.frombuffer(binary, dtype=np.uint8).reshape(text["shape"])
        else:
            self.img = _decode_image(binary)


# Check if 'cuda-python' and 'torch' are available
enable_torch_image = True
try:
    # import cuda.bindings as cuda
    from cuda.bindings import driver
except ImportError:
    print("WARNING: 'cuda-python' not found. Stubbing 'TorchImage' with 'NumpyImage'.")
    enable_torch_image = False

try:
    import torch
except ImportError:
    print("WARNING: 'torch' not found. Stubbing 'TorchImage' with 'NumpyImage'.")
    enable_torch_image = False
else:
    if not torch.cuda.is_available():
        print("WARNING: 'torch' is not compiled with CUDA support. Stubbing 'TorchImage' with 'NumpyImage'.")
        enable_torch_image = False

if enable_torch_image:
    class TorchImage(Image):
        """ Image viewer where the image to be shown comes from Torch tensor **on the GPU**. """
        _cuda_resource = None

        def _upload(self):
            img = self.img
            assert img.is_cuda, "'img' is not the GPU."
            if img.dtype != torch.uint8:
                img = (torch.clip(img, 0, 1) * 255).byte()

            if img.shape[1] != self.texture.res_x or img.shape[0] != self.texture.res_y:
                if self._cuda_resource is not None:
                    checkCudaErrors(driver.cuGraphicsUnregisterResource(self._cuda_resource))
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.shape[1], img.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, img.cpu().numpy())
                self._cuda_resource = checkCudaErrors(driver.cuGraphicsGLRegisterImage(self.texture.id, GL_TEXTURE_2D, driver.CUgraphicsRegisterFlags.CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD))
                self.texture.res_x = img.shape[1]
                self.texture.res_y = img.shape[0]
            else:
                # For some reason the copy only works for 4 byte pixels
                if img.shape[-1] == 3:
                    img = torch.cat([img, 255 * torch.ones((img.shape[0], img.shape[1], 1), device=img.device, dtype=torch.uint8)], -1)
                # Copy to OpenGL texture
                checkCudaErrors(driver.cuGraphicsMapResources(1, self._cuda_resource, 0))
                cuda_array = checkCudaErrors(driver.cuGraphicsSubResourceGetMappedArray(self._cuda_resource, 0, 0))
                copy_params = driver.CUDA_MEMCPY3D()
                copy_params.srcMemoryType = driver.CUmemorytype.CU_MEMORYTYPE_DEVICE
                copy_params.srcDevice = img.data_ptr()
                copy_params.srcPitch = self.texture.res_x * 4
                copy_params.dstMemoryType = driver.CUmemorytype.CU_MEMORYTYPE_ARRAY
                copy_params.dstArray = cuda_array
                copy_params.WidthInBytes = self.texture.res_x * 4
                copy_params.Height = self.texture.res_y
                copy_params.Depth = 1
                checkCudaErrors(driver.cuMemcpy3D(copy_params))
                checkCudaErrors(driver.cuGraphicsUnmapResources(1, self._cuda_resource, 0))

        def destroy(self):
            """ Delete the 'cuda_resource' then delete the OpenGL texture. """
            if self._cuda_resource is not None:
                checkCudaErrors(driver.cuGraphicsUnregisterResource(self._cuda_resource))
            super().destroy()
        
        def server_send(self):
            if not self.step_called:
                return None, None
            img = self.img
            if img.dtype != torch.uint8:
                img = (torch.clip(img, 0, 1) * 255).byte()
            self.step_called = False
            if self._compression is not None:
                binary = _encode_image(img.contiguous().cpu().numpy(), self._compression, self._quality)
                return binary, {"format": self._compression}
            return memoryview(img.contiguous().flatten().cpu().numpy()), {"shape": tuple(img.shape), "format": "raw"}

        def client_recv(self, binary, text):
            if text.get("format", "raw") == "raw":
                img = np.frombuffer(binary, dtype=np.uint8).reshape(text["shape"])
            else:
                img = _decode_image(binary)
            self.img = torch.from_numpy(img).to(0)

else:
    class TorchImage(NumpyImage):
        # Update the step function to convert the input to a tensor to numpy array
        def step(self, img):
            if not isinstance(img, np.ndarray):
                # A tensor
                img = img.detach().cpu().numpy()
            # Otherwise it's already a numpy array
            self.img = img